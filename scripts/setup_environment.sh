# scripts/setup_environment.sh

# setup script for the Fake News Detection project
# this script creates a virtual environment and installs dependencies

# set up colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # no Color

# print message with a colored prefix
function print_message() {
    local color=$1
    local prefix=$2
    local message=$3
    echo -e "${color}${prefix}:${NC} ${message}"
}

function info() {
    print_message "${BLUE}" "INFO" "$1"
}

function success() {
    print_message "${GREEN}" "SUCCESS" "$1"
}

function warning() {
    print_message "${YELLOW}" "WARNING" "$1"
}

function error() {
    print_message "${RED}" "ERROR" "$1"
}

# check Python version
info "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    error "Python 3.9+ is required. Found: Python $PYTHON_VERSION"
    exit 1
fi
success "Python $PYTHON_VERSION detected"

# create necessary directories
info "Creating project directories..."
mkdir -p data/raw/{text,images,metadata}
mkdir -p data/processed/{text,images,metadata}
mkdir -p data/external
mkdir -p models/saved
mkdir -p models/output
mkdir -p logs
mkdir -p notebooks/{exploratory,experiments}
success "Directories created"

# create virtual environment
info "Creating virtual environment..."
if [ -d "venv" ]; then
    warning "Virtual environment already exists. Skipping creation."
else
    python3 -m venv venv
    success "Virtual environment created"
fi

# activate virtual environment
info "Activation virtual environment..."
source venv/bin/activate
success "Virtual environment activated"

# install dependencies
info "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# check if GPU is available for PyTorch
info "Checking for GPU availability..."
python3 -c "import torch; print('GPU available: ' + str(torch.cuda.is_available()))"
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    success "GPU is available for PyTorch"
else
    warning "No GPU detected for PyTorch. Training will use CPU only."
fi

# install pre-commit hooks
info "Setting up pre-commit hooks..."
if command -v pre-commit &> /dev/null; then
    pre-commit install
    success "Pre-commit hooks installed"
else
    warning "Pre commit not found. Skipping hook installation."
fi

# download spaCy model
info "Downloading spaCy English model..."
python3 -m spacy download en_core_web_sm
success "spaCy model downloaded"

# download NLTK data
info "Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('vader_lexicon')"
success "NLTK data downloaded"

# create .env file
info "Creating .env file..."
if [ -f ".env" ]; then
    warning ".env file already exists. Skipping creation."
else
    cat > .env << EOL
# environment variables for the Fake News Detection project

# paths
DATA_DIR=./data
MODELS_DIR=./models
LOGS_DIR=./logs

# gpu settings
USE_GPU=false
CUDA_VISIBLE_DEVICES=0
EOL
    success ".env file created"
fi

# final message
echo ""
echo -e "${GREEN}===============================================${NC}"
echo -e "${GREEN}Setup completed successfully!${NC}"
echo -e "${GREEN}===============================================${NC}"
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To download datasets, run:"
echo "  python scripts/data_collection/download_datasets.py"
echo ""
echo "Next steps:"
echo "1. Update the .env file with your configuration"
echo "2. Download the datasets"
echo "3. Explore the data in notebooks/exploratory"
echo ""
echo "Happy coding!"