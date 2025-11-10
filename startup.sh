#!/bin/bash

echo "Setting up Task Offloading Algorithm Comparison Framework"
echo "=========================================================="

# Create Python virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install graphviz system package (Ubuntu/Debian)
echo "Installing graphviz system package..."
sudo apt-get update
sudo apt-get install -y graphviz

# Create necessary directories
echo "Creating directories..."
mkdir -p results/{plots,logs,models}
mkdir -p data/test_graphs

# Verify installation
echo "Verifying installation..."
python -c "import torch; import numpy; import gym; print('✓ Core packages installed')"
python -c "import networkx; import pydotplus; print('✓ Graph packages installed')"
python -c "import yaml; import matplotlib; print('✓ Utility packages installed')"

echo ""
echo "Setup complete!"
echo "To activate the environment, run: source venv/bin/activate"
echo "To run tests, use: python main_training.py --help"

