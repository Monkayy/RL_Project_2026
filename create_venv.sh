#!/usr/bin/env bash
set -e

echo "Creating the virtual environment..."
python3 -m venv .venv
echo "Activating the virtual environment..."
source .venv/bin/activate

echo "Installing the required libraries. Hold on..."
pip install -r requirements.txt

echo "Virtual environment ready and activated!"