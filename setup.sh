#!/usr/bin/env bash
# Convenience setup script for the UTTHAAN prototype.
set -e

echo "Installing system dependency: tesseract-ocr ..."
if command -v apt-get >/dev/null; then
    sudo apt-get update -qq && sudo apt-get install -y -qq tesseract-ocr
else
    echo "Please install tesseract-ocr manually for your OS (see README)."
fi

echo "Installing Python dependencies ..."
pip install -r requirements.txt

echo "Done. Run the demo with:"
echo "  python -m src.pipeline      # generates data + runs the full pipeline"
echo "  python -m src.dashboard     # launches the live dashboard at http://localhost:5050"
