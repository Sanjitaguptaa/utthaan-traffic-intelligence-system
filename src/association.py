"""
Cross-Camera Intelligence layer: matches per-camera vehicle observations into
continuous city-wide trajectories, using the same signal set the deck lists
(Plate Similarity, Appearance Similarity, Timestamp Compatibility, Camera
Location / Direction Consistency, Expected Travel Time) and produces the
"Dynamic Trajectory Confidence Score" (High >=80%, Probable 50-79%,
Review Required <50%).
"""
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import config
from .reid import similarity as appearance_similarity


def plate_similarity(plate_a, plate_b):
    if not plate_a or not plate_b:
        return None  # unknown, not zero — handled by weight renormalization
    return SequenceMatcher(None, plate_a, plate_b).ratio()


def timestamp_compatibility(obs_a, obs_b, cam_a, cam_b):
    distance_m = config.CAMERAS[cam_b]["position_m"] - config.CAMERAS[cam_a]["position_m"]
    dt = obs_b.t_mid - obs_a.t_mid
    if dt <= 0 or distance_m <= 0:
        return 0.0
    implied_speed_kmh = (distance_m / dt) * 3.6
    lo, hi = config.EXPECTED_SPEED_MIN_KMH, config.EXPECTED_SPEED_MAX_KMH
    if lo <= implied_speed_kmh <= hi:
        return 1.0
    # Graceful falloff outside the plausible speed band rather than a hard cutoff
    margin = min(abs(implied_speed_kmh - lo), abs(implied_speed_kmh - hi))
    return max(0.0, 1.0 - margin / 40.0)


def link_confidence(obs_a, obs_b, cam_a, cam_b):
    weights = dict(config.ASSOCIATION_WEIGHTS)
    scores = {
        "plate_similarity": plate_similarity(obs_a.plate, obs_b.plate),
        "appearance_similarity": appearance_similarity(obs_a.appearance_vec, obs_b.appearance_vec),
        "timestamp_compatibility": timestamp_compatibility(obs_a, obs_b, cam_a, cam_b),
        "direction_consistency": 1.0,  # only adjacent-in-order cameras are ever paired
    }
    total_weight = 0.0
    total_score = 0.0
    for key, val in scores.items():
        if val is None:
            continue  # e.g. plate unreadable on one side -> drop & renormalize
        total_weight += weights[key]
        total_score += weights[key] * val
    if total_weight == 0:
        return 0.0, scores
    return total_score / total_weight, scores


def confidence_label(score):
    if score >= config.HIGH_CONFIDENCE_THRESHOLD:
        return "High Confidence"
    if score >= config.PROBABLE_THRESHOLD:
        return "Probable"
    return "Review Required"


@dataclass
class TrajectoryLeg:
    camera_a: str
    camera_b: str
    obs_a: "object"
    obs_b: "object"
    confidence: float
    score_breakdown: dict


@dataclass
class Trajectory:
    vehicle_key: str
    plate: Optional[str]
    legs: List[TrajectoryLeg]
    camera_sequence: List[str]
    overall_confidence: float
    label: str


def match_adjacent_cameras(obs_a_list, obs_b_list, cam_a, cam_b):
    """Hungarian-assignment matching between two adjacent cameras' observations."""
    if not obs_a_list or not obs_b_list:
        return []
    cost = np.zeros((len(obs_a_list), len(obs_b_list)))
    conf_matrix = {}
    for i, oa in enumerate(obs_a_list):
        for j, ob in enumerate(obs_b_list):
            conf, breakdown = link_confidence(oa, ob, cam_a, cam_b)
            conf_matrix[(i, j)] = (conf, breakdown)
            cost[i, j] = 1 - conf

    row_idx, col_idx = linear_sum_assignment(cost)
    legs = []
    for r, c in zip(row_idx, col_idx):
        conf, breakdown = conf_matrix[(r, c)]
        if conf >= config.PROBABLE_THRESHOLD:
            legs.append(TrajectoryLeg(cam_a, cam_b, obs_a_list[r], obs_b_list[c], conf, breakdown))
    return legs


def build_trajectories(observations_by_camera):
    """Chains adjacent-camera matches across the full camera order into
    end-to-end city-wide vehicle trajectories."""
    order = config.CAMERA_ORDER
    all_legs = {}
    for cam_a, cam_b in zip(order, order[1:]):
        all_legs[(cam_a, cam_b)] = match_adjacent_cameras(
            observations_by_camera.get(cam_a, []),
            observations_by_camera.get(cam_b, []),
            cam_a, cam_b,
        )

    # Chain legs starting from every observation at the first camera, and also
    # allow trajectories that start mid-corridor (vehicle entered view late).
    chains = []
    used_as_b = set()
    for start_cam_idx, start_cam in enumerate(order):
        for obs in observations_by_camera.get(start_cam, []):
            if id(obs) in used_as_b:
                continue
            chain = [obs]
            chain_cams = [start_cam]
            legs_in_chain = []
            cur_cam_idx = start_cam_idx
            cur_obs = obs
            while cur_cam_idx < len(order) - 1:
                cam_a, cam_b = order[cur_cam_idx], order[cur_cam_idx + 1]
                leg = next((l for l in all_legs[(cam_a, cam_b)] if l.obs_a is cur_obs), None)
                if leg is None:
                    break
                legs_in_chain.append(leg)
                chain.append(leg.obs_b)
                chain_cams.append(cam_b)
                used_as_b.add(id(leg.obs_b))
                cur_obs = leg.obs_b
                cur_cam_idx += 1
            chains.append((chain, chain_cams, legs_in_chain))

    trajectories = []
    for i, (chain, chain_cams, legs) in enumerate(chains):
        plate_votes = [o.plate for o in chain if o.plate]
        plate = max(set(plate_votes), key=plate_votes.count) if plate_votes else None
        if legs:
            overall_conf = float(np.mean([l.confidence for l in legs]))
        else:
            overall_conf = 1.0 if chain[0].plate else 0.5  # single-camera sighting only
        trajectories.append(Trajectory(
            vehicle_key=plate or f"UNKNOWN-{chain_cams[0]}-{chain[0].track_id}",
            plate=plate,
            legs=legs,
            camera_sequence=chain_cams,
            overall_confidence=overall_conf,
            label=confidence_label(overall_conf),
        ))
    return trajectories
