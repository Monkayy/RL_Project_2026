#!/usr/bin/env bash
set -e

if [ -d ".venv" ]; then
    echo "Virtual environment already exists, skipping creation..."
else
    echo "Creating the virtual environment..."
    python3 -m venv .venv
fi

echo "Activating the virtual environment..."
source .venv/bin/activate

echo "Updating pip..."
pip install --upgrade pip

echo "Installing the required libraries. Hold on..."
pip install -r requirements.txt

echo "#########################################"
echo "          You're ready to go!"
echo "#########################################"