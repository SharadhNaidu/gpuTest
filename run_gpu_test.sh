#!/bin/bash

#################################################################
# GPU Testing Suite - Automated Setup and Run Script
# MiPhi GPU Facility - College GPU Testing
#
# This script handles everything automatically:
# - Works with Python 3.6 to 3.12+
# - Multiple fallback methods for package installation
# - Proper PyTorch version and CUDA compatibility checking
# - No manual setup required
#
# Usage: ./run_gpu_test.sh
#################################################################

# Minimum required versions
MIN_PYTORCH_VERSION="2.0.0"
MIN_PYTHON_VERSION="3.7"

# Colors for output (with fallback for non-color terminals)
if [ -t 1 ] && [ -n "$TERM" ] && [ "$TERM" != "dumb" ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    CYAN=''
    NC=''
fi

# Print banner
echo ""
echo -e "${CYAN}+---------------------------------------------------------------+${NC}"
echo -e "${CYAN}|         ${GREEN}GPU TESTING SUITE - MiPhi GPU Facility${CYAN}              |${NC}"
echo -e "${CYAN}|         ${YELLOW}Comprehensive GPU Analysis & Testing${CYAN}               |${NC}"
echo -e "${CYAN}+---------------------------------------------------------------+${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Function to print status messages
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

print_status "Starting GPU Testing Suite..."
print_status "Working directory: $SCRIPT_DIR"
echo ""

# ===============================================
# Step 1: Check for NVIDIA GPU and drivers
# ===============================================
print_status "Checking for NVIDIA GPU..."

if ! command_exists nvidia-smi; then
    print_error "nvidia-smi not found!"
    print_error "Please install NVIDIA drivers first."
    echo ""
    echo "Installation instructions:"
    echo "  Ubuntu/Debian: sudo apt install nvidia-driver-XXX"
    echo "  RHEL/CentOS:   sudo dnf install nvidia-driver"
    echo "  Or download from: https://www.nvidia.com/drivers"
    exit 1
fi

# Test nvidia-smi
if ! nvidia-smi > /dev/null 2>&1; then
    print_error "nvidia-smi failed to run!"
    print_error "GPU drivers may not be properly installed or GPU may not be available."
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1)
print_success "Found GPU: ${GPU_NAME}"
print_success "Driver Version: ${DRIVER_VERSION}"
echo ""

# ===============================================
# Step 2: Find Python 3
# ===============================================
print_status "Checking for Python 3..."

PYTHON_CMD=""

# Try different Python commands in order of preference
for cmd in python3 python python3.12 python3.11 python3.10 python3.9 python3.8 python3.7 python3.6; do
    if command_exists "$cmd"; then
        # Check if it's Python 3
        VERSION=$($cmd -c "import sys; print(sys.version_info.major)" 2>/dev/null)
        if [ "$VERSION" = "3" ]; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    print_error "Python 3 not found!"
    echo ""
    echo "Please install Python 3:"
    echo "  Ubuntu/Debian: sudo apt install python3 python3-pip"
    echo "  RHEL/CentOS:   sudo dnf install python3 python3-pip"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
PYTHON_MINOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.minor)" 2>/dev/null)
print_success "Found: ${PYTHON_VERSION}"
echo ""

# ===============================================
# Step 3: Setup Python environment and packages
# ===============================================
VENV_DIR="$SCRIPT_DIR/.gpu_test_venv"
USE_VENV=false
INSTALL_SUCCESS=false

print_status "Setting up Python environment..."

# Function to check if required packages are available
check_packages() {
    local python_cmd="$1"
    $python_cmd -c "import numpy, matplotlib, pandas, psutil" 2>/dev/null
    return $?
}

# Function to install packages
install_packages() {
    local pip_cmd="$1"
    local extra_args="$2"
    
    print_status "Installing packages with: $pip_cmd $extra_args"
    
    # Install core packages
    $pip_cmd install $extra_args numpy matplotlib pandas psutil 2>&1 | tail -5
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        return 0
    fi
    return 1
}

# Function to install PyTorch
install_pytorch() {
    local pip_cmd="$1"
    local extra_args="$2"
    local python_cmd="$3"
    local force_install="$4"
    
    # Get system CUDA version
    SYSTEM_CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' 2>/dev/null || echo "")
    SYSTEM_CUDA_MAJOR=$(echo "$SYSTEM_CUDA_VERSION" | cut -d'.' -f1 2>/dev/null || echo "")
    
    if [ -z "$SYSTEM_CUDA_MAJOR" ]; then
        print_warning "Could not detect system CUDA version"
        return 1
    fi
    
    # Check if PyTorch is already installed
    if $python_cmd -c "import torch" 2>/dev/null; then
        INSTALLED_TORCH_VERSION=$($python_cmd -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null)
        TORCH_CUDA_AVAILABLE=$($python_cmd -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
        TORCH_CUDA_VERSION=$($python_cmd -c "import torch; print(torch.version.cuda if torch.version.cuda else 'None')" 2>/dev/null)
        
        print_status "Found existing PyTorch: ${INSTALLED_TORCH_VERSION}"
        print_status "  CUDA available: ${TORCH_CUDA_AVAILABLE}"
        print_status "  PyTorch CUDA version: ${TORCH_CUDA_VERSION}"
        print_status "  System CUDA version: ${SYSTEM_CUDA_VERSION}"
        
        # Check if version is sufficient
        TORCH_OK=$($python_cmd -c "
from packaging import version
import torch
v = torch.__version__.split('+')[0]
print('yes' if version.parse(v) >= version.parse('$MIN_PYTORCH_VERSION') else 'no')
" 2>/dev/null || echo "no")
        
        # Check CUDA compatibility (PyTorch CUDA major should match or be close to system CUDA major)
        TORCH_CUDA_MAJOR=$(echo "$TORCH_CUDA_VERSION" | cut -d'.' -f1 2>/dev/null || echo "")
        
        CUDA_COMPATIBLE="no"
        if [ "$TORCH_CUDA_AVAILABLE" = "True" ] && [ -n "$TORCH_CUDA_MAJOR" ] && [ -n "$SYSTEM_CUDA_MAJOR" ]; then
            # Allow CUDA version difference of 1 major version
            DIFF=$((SYSTEM_CUDA_MAJOR - TORCH_CUDA_MAJOR))
            if [ "$DIFF" -ge -1 ] && [ "$DIFF" -le 2 ]; then
                CUDA_COMPATIBLE="yes"
            fi
        fi
        
        if [ "$TORCH_OK" = "yes" ] && [ "$CUDA_COMPATIBLE" = "yes" ] && [ "$force_install" != "true" ]; then
            print_success "PyTorch ${INSTALLED_TORCH_VERSION} is compatible (CUDA ${TORCH_CUDA_VERSION})"
            return 0
        elif [ "$TORCH_CUDA_AVAILABLE" != "True" ]; then
            print_warning "PyTorch installed but CUDA not available"
            print_status "Will reinstall PyTorch with CUDA support..."
        elif [ "$TORCH_OK" != "yes" ]; then
            print_warning "PyTorch version ${INSTALLED_TORCH_VERSION} is below minimum ${MIN_PYTORCH_VERSION}"
            print_status "Will upgrade PyTorch..."
        elif [ "$CUDA_COMPATIBLE" != "yes" ]; then
            print_warning "PyTorch CUDA version (${TORCH_CUDA_VERSION}) may not be compatible with system CUDA (${SYSTEM_CUDA_VERSION})"
            print_status "Will reinstall PyTorch with matching CUDA version..."
        fi
    fi
    
    print_status "Detected system CUDA version: ${SYSTEM_CUDA_VERSION}"
    print_status "Installing PyTorch (this may take several minutes)..."
    
    # First install packaging module for version comparison
    $pip_cmd install $extra_args packaging 2>/dev/null || true
    
    # Uninstall existing torch to avoid conflicts
    $pip_cmd uninstall -y torch torchvision 2>/dev/null || true
    
    if [ "$SYSTEM_CUDA_MAJOR" -ge 12 ]; then
        print_status "Installing PyTorch for CUDA 12.x..."
        $pip_cmd install $extra_args torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -5
    elif [ "$SYSTEM_CUDA_MAJOR" -ge 11 ]; then
        print_status "Installing PyTorch for CUDA 11.x..."
        $pip_cmd install $extra_args torch torchvision --index-url https://download.pytorch.org/whl/cu118 2>&1 | tail -5
    else
        print_status "Installing PyTorch for older CUDA..."
        $pip_cmd install $extra_args torch torchvision 2>&1 | tail -5
    fi
    
    # Verify installation
    if $python_cmd -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        TORCH_VERSION=$($python_cmd -c "import torch; print(torch.__version__)" 2>/dev/null)
        TORCH_CUDA=$($python_cmd -c "import torch; print(torch.version.cuda)" 2>/dev/null)
        print_success "PyTorch ${TORCH_VERSION} installed with CUDA ${TORCH_CUDA}"
        return 0
    else
        print_warning "PyTorch CUDA not available after installation - tests will be limited"
        return 1
    fi
}

# ===============================================
# Method 1: Try using existing virtual environment
# ===============================================
if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/python" ]; then
    print_status "Found existing virtual environment..."
    if check_packages "$VENV_DIR/bin/python"; then
        print_success "Using existing virtual environment"
        PYTHON_CMD="$VENV_DIR/bin/python"
        PIP_CMD="$VENV_DIR/bin/pip"
        USE_VENV=true
        INSTALL_SUCCESS=true
    else
        print_warning "Existing venv incomplete, will recreate..."
        rm -rf "$VENV_DIR"
    fi
fi

# ===============================================
# Method 2: Try creating a new virtual environment
# ===============================================
if [ "$INSTALL_SUCCESS" = false ]; then
    print_status "Attempting to create virtual environment..."
    
    # Try to create venv
    if $PYTHON_CMD -m venv "$VENV_DIR" 2>/dev/null; then
        print_success "Virtual environment created"
        
        PYTHON_CMD="$VENV_DIR/bin/python"
        PIP_CMD="$VENV_DIR/bin/pip"
        USE_VENV=true
        
        # Upgrade pip
        $PIP_CMD install --upgrade pip 2>/dev/null || true
        
        # Install packages
        if install_packages "$PIP_CMD" ""; then
            print_success "Core packages installed in venv"
            INSTALL_SUCCESS=true
        else
            print_warning "Failed to install packages in venv"
            rm -rf "$VENV_DIR"
            USE_VENV=false
            # Reset Python command
            for cmd in python3 python; do
                if command_exists "$cmd"; then
                    VERSION=$($cmd -c "import sys; print(sys.version_info.major)" 2>/dev/null)
                    if [ "$VERSION" = "3" ]; then
                        PYTHON_CMD="$cmd"
                        break
                    fi
                fi
            done
        fi
    else
        print_warning "Could not create virtual environment"
        print_status "Trying alternative installation methods..."
    fi
fi

# ===============================================
# Method 3: Try pip install --user
# ===============================================
if [ "$INSTALL_SUCCESS" = false ]; then
    print_status "Trying user-level package installation..."
    
    PIP_CMD="$PYTHON_CMD -m pip"
    
    if install_packages "$PIP_CMD" "--user"; then
        if check_packages "$PYTHON_CMD"; then
            print_success "Packages installed with --user"
            INSTALL_SUCCESS=true
        fi
    fi
fi

# ===============================================
# Method 4: Try pip install --break-system-packages (Python 3.11+)
# ===============================================
if [ "$INSTALL_SUCCESS" = false ] && [ "$PYTHON_MINOR" -ge 11 ]; then
    print_status "Trying with --break-system-packages flag (Python 3.11+)..."
    
    if install_packages "$PIP_CMD" "--user --break-system-packages"; then
        if check_packages "$PYTHON_CMD"; then
            print_success "Packages installed with --break-system-packages"
            INSTALL_SUCCESS=true
        fi
    fi
fi

# ===============================================
# Method 5: Try system pip directly
# ===============================================
if [ "$INSTALL_SUCCESS" = false ]; then
    print_status "Trying direct pip installation..."
    
    for pip_cmd in pip3 pip; do
        if command_exists "$pip_cmd"; then
            $pip_cmd install --user numpy matplotlib pandas psutil 2>&1 | tail -3
            if check_packages "$PYTHON_CMD"; then
                print_success "Packages installed with $pip_cmd"
                INSTALL_SUCCESS=true
                break
            fi
            
            # Try with --break-system-packages
            $pip_cmd install --user --break-system-packages numpy matplotlib pandas psutil 2>&1 | tail -3
            if check_packages "$PYTHON_CMD"; then
                print_success "Packages installed with $pip_cmd --break-system-packages"
                INSTALL_SUCCESS=true
                break
            fi
        fi
    done
fi

# ===============================================
# Method 6: Try with sudo (last resort)
# ===============================================
if [ "$INSTALL_SUCCESS" = false ]; then
    print_warning "All user-level installations failed."
    print_status "Attempting system-level installation (may require password)..."
    
    if command_exists apt-get; then
        sudo apt-get update -qq 2>/dev/null
        sudo apt-get install -y -qq python3-numpy python3-matplotlib python3-pandas python3-psutil python3-pip python3-venv 2>/dev/null
        if check_packages "$PYTHON_CMD"; then
            print_success "Packages installed via apt"
            INSTALL_SUCCESS=true
        fi
    elif command_exists dnf; then
        sudo dnf install -y -q python3-numpy python3-matplotlib python3-pandas python3-psutil python3-pip 2>/dev/null
        if check_packages "$PYTHON_CMD"; then
            print_success "Packages installed via dnf"
            INSTALL_SUCCESS=true
        fi
    elif command_exists yum; then
        sudo yum install -y -q python3-numpy python3-matplotlib python3-pandas python3-psutil python3-pip 2>/dev/null
        if check_packages "$PYTHON_CMD"; then
            print_success "Packages installed via yum"
            INSTALL_SUCCESS=true
        fi
    fi
fi

# ===============================================
# Method 7: Try creating venv after installing python3-venv
# ===============================================
if [ "$INSTALL_SUCCESS" = false ]; then
    print_status "Trying to install python3-venv and retry..."
    
    if command_exists apt-get; then
        sudo apt-get install -y python3-venv 2>/dev/null
    fi
    
    if $PYTHON_CMD -m venv "$VENV_DIR" 2>/dev/null; then
        PYTHON_CMD="$VENV_DIR/bin/python"
        PIP_CMD="$VENV_DIR/bin/pip"
        USE_VENV=true
        
        $PIP_CMD install --upgrade pip 2>/dev/null || true
        
        if install_packages "$PIP_CMD" ""; then
            print_success "Core packages installed in venv (after installing python3-venv)"
            INSTALL_SUCCESS=true
        fi
    fi
fi

# ===============================================
# Final check
# ===============================================
if [ "$INSTALL_SUCCESS" = false ]; then
    print_error "Failed to install required Python packages!"
    echo ""
    echo "Please try manually:"
    echo "  1. Install venv support:"
    echo "     sudo apt install python3-venv python3-pip"
    echo ""
    echo "  2. Create a virtual environment:"
    echo "     python3 -m venv .venv && source .venv/bin/activate"
    echo ""
    echo "  3. Install packages:"
    echo "     pip install numpy matplotlib pandas psutil torch"
    echo ""
    echo "  4. Run the test:"
    echo "     python gpu_test.py"
    exit 1
fi

# Verify packages one more time
if ! check_packages "$PYTHON_CMD"; then
    print_error "Package verification failed!"
    exit 1
fi

print_success "All required packages are available"
echo ""

# ===============================================
# Install PyTorch (optional but recommended)
# ===============================================
print_status "Setting up PyTorch for GPU tests..."

if [ "$USE_VENV" = true ]; then
    install_pytorch "$PIP_CMD" "" "$PYTHON_CMD" "false"
else
    # Try different methods for PyTorch
    install_pytorch "$PYTHON_CMD -m pip" "--user" "$PYTHON_CMD" "false" || \
    install_pytorch "$PYTHON_CMD -m pip" "--user --break-system-packages" "$PYTHON_CMD" "false" || \
    print_warning "Could not install PyTorch - some tests will be skipped"
fi

# Try to install pynvml
print_status "Installing optional monitoring package (pynvml)..."
if [ "$USE_VENV" = true ]; then
    $PIP_CMD install pynvml 2>/dev/null && print_success "pynvml installed" || print_warning "pynvml not available"
else
    $PYTHON_CMD -m pip install --user pynvml 2>/dev/null && print_success "pynvml installed" || \
    $PYTHON_CMD -m pip install --user --break-system-packages pynvml 2>/dev/null && print_success "pynvml installed" || \
    print_warning "pynvml not available - using nvidia-smi instead"
fi

echo ""

# ===============================================
# Step 4: Check if test script exists
# ===============================================
print_status "Checking for test script..."

if [ ! -f "gpu_test.py" ]; then
    print_error "gpu_test.py not found in current directory!"
    print_error "Please ensure the test script is in: $SCRIPT_DIR"
    exit 1
fi

print_success "Test script found: gpu_test.py"
echo ""

# ===============================================
# Step 5: Run the GPU test
# ===============================================
echo -e "${CYAN}+---------------------------------------------------------------+${NC}"
echo -e "${CYAN}|              ${GREEN}STARTING GPU TESTING SUITE${CYAN}                     |${NC}"
echo -e "${CYAN}+---------------------------------------------------------------+${NC}"
echo ""

# Show which Python we're using
print_status "Using Python: $PYTHON_CMD"
$PYTHON_CMD --version
echo ""

# Run the test
$PYTHON_CMD gpu_test.py

TEST_EXIT_CODE=$?

echo ""

# ===============================================
# Step 6: Display results
# ===============================================
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${CYAN}+---------------------------------------------------------------+${NC}"
    echo -e "${CYAN}|              ${GREEN}GPU TESTING COMPLETED SUCCESSFULLY${CYAN}             |${NC}"
    echo -e "${CYAN}+---------------------------------------------------------------+${NC}"
    echo ""
    
    print_success "All tests completed!"
    echo ""
    
    # List generated files
    echo -e "${BLUE}Generated Reports:${NC}"
    
    if [ -f "gpu_report.png" ]; then
        echo -e "  ${GREEN}[OK]${NC} gpu_report.png     - Visual analytics report"
    fi
    
    if [ -f "gpu_report.json" ]; then
        echo -e "  ${GREEN}[OK]${NC} gpu_report.json    - Machine-readable data"
    fi
    
    if [ -d "gpu_test_results" ]; then
        echo -e "  ${GREEN}[OK]${NC} gpu_test_results/  - Timestamped archives"
    fi
    
    echo ""
    echo -e "${BLUE}Quick Actions:${NC}"
    echo "  - View image report:  xdg-open gpu_report.png  (or open manually)"
    echo "  - View JSON data:     cat gpu_report.json"
    echo ""
    
else
    echo -e "${CYAN}+---------------------------------------------------------------+${NC}"
    echo -e "${CYAN}|              ${RED}GPU TESTING ENCOUNTERED ERRORS${CYAN}                 |${NC}"
    echo -e "${CYAN}+---------------------------------------------------------------+${NC}"
    echo ""
    
    print_error "Some tests may have failed. Check the output above for details."
    echo ""
fi

echo -e "${CYAN}---------------------------------------------------------------${NC}"
echo -e "${GREEN}Thank you for using the MiPhi GPU Testing Suite!${NC}"
echo -e "${CYAN}---------------------------------------------------------------${NC}"
echo ""

exit $TEST_EXIT_CODE
