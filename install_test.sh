# Default Python version
PYTHON_VERSION=python=3.10  # für Kompatibilität mit Pytorch Version und cu117

# Default PyTorch version
# Available options are 1.4.0 and 2.0.1
PYTORCH_VERSION=${1:-"2.0.1"}

# Default Torch Geometric version
TORCH_GEOMETRIC_VERSION=2.3.1

# Set CUDA variable (defaults to cpu if no argument is provided to the script)
# Available options for PyTorch 2.0.1 are cpu, and cu117
CUDA_VERSION=${2:-"cpu"}

# Define the Conda environment name
CONDA_ENV_NAME=gnn-comparison

echo "Using Python version: ${PYTHON_VERSION}"
echo "Creating Conda environment: ${CONDA_ENV_NAME}"
echo "Installing PyTorch ${PYTORCH_VERSION} with CUDA ${CUDA_VERSION} support"
echo "Torch Geometric version: ${TORCH_GEOMETRIC_VERSION}"

# Create the Conda environment
conda create -y -n ${CONDA_ENV_NAME} ${PYTHON_VERSION}

# Activate the Conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV_NAME}

# Install build tools
pip install --upgrade pip
pip install build wheel

# Install PyTorch
if [[ "$CUDA_VERSION" == "cpu" ]]; then
  pip install torch==${PYTORCH_VERSION} --extra-index-url https://download.pytorch.org/whl/cpu
elif [[ "$CUDA_VERSION" == 'cu116' ]]; then
  pip install torch==${PYTORCH_VERSION} --extra-index-url https://download.pytorch.org/whl/cu116
elif [[ "$CUDA_VERSION" == 'cu117' ]]; then
  pip install torch==${PYTORCH_VERSION} --extra-index-url https://download.pytorch.org/whl/cu117
elif [[ "$CUDA_VERSION" == 'cu118' ]]; then
  pip install torch==${PYTORCH_VERSION} --extra-index-url https://download.pytorch.org/whl/cu118
fi

# Install Torch Geometric
pip install torch-geometric==${TORCH_GEOMETRIC_VERSION}

# Install additional dependencies
pip install PyYAML==6.0.1

echo "Installation complete. To activate the Conda environment, use: conda activate ${CONDA_ENV_NAME}"