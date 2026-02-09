#!/usr/bin/env python3
"""
AI Desktop Assessment Suite - One Command Runner
Works on: Windows, Linux, macOS
Usage: python run.py
"""

import os
import sys
import subprocess
import platform

def main():
    print("=" * 50)
    print("  AI Desktop Assessment Suite - Setup & Run")
    print("=" * 50)
    print()
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Determine OS-specific paths
    is_windows = platform.system() == "Windows"
    venv_dir = os.path.join(script_dir, ".venv")
    
    if is_windows:
        python_exe = os.path.join(venv_dir, "Scripts", "python.exe")
        pip_exe = os.path.join(venv_dir, "Scripts", "pip.exe")
    else:
        python_exe = os.path.join(venv_dir, "bin", "python")
        pip_exe = os.path.join(venv_dir, "bin", "pip")
    
    # Step 1: Create virtual environment if not exists
    if not os.path.exists(python_exe):
        print("[1/3] Creating virtual environment...")
        try:
            subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
            print("      ✓ Virtual environment created")
        except subprocess.CalledProcessError as e:
            print(f"      ✗ Failed to create venv: {e}")
            print("      Trying without venv...")
            python_exe = sys.executable
            pip_exe = None
    else:
        print("[1/3] Virtual environment exists ✓")
    
    # Step 2: Install dependencies
    print("[2/3] Installing dependencies...")
    packages = ["numpy", "matplotlib", "pandas", "psutil"]
    
    try:
        if pip_exe and os.path.exists(pip_exe):
            # Use venv pip
            subprocess.run(
                [pip_exe, "install", "-q", "--upgrade", "pip"],
                capture_output=True
            )
            subprocess.run(
                [pip_exe, "install", "-q"] + packages,
                check=True,
                capture_output=True
            )
        else:
            # Fallback to system pip
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q"] + packages,
                check=True,
                capture_output=True
            )
        print("      ✓ Dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"      ✗ pip install failed, trying with --user flag...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "--user"] + packages,
                check=True,
                capture_output=True
            )
            print("      ✓ Dependencies installed (user mode)")
            python_exe = sys.executable
        except:
            print("      ✗ Could not install packages automatically")
            print(f"      Please run: pip install {' '.join(packages)}")
            sys.exit(1)
    
    # Step 3: Run the test
    print("[3/3] Running AI Desktop Assessment...")
    print()
    print("-" * 50)
    print()
    
    test_script = os.path.join(script_dir, "gpu_test.py")
    
    # Use the venv python if available
    if os.path.exists(python_exe):
        result = subprocess.run([python_exe, test_script])
    else:
        result = subprocess.run([sys.executable, test_script])
    
    print()
    print("-" * 50)
    
    if result.returncode == 0:
        print()
        print("✓ Assessment complete! Check the output files:")
        # List generated files
        for f in os.listdir(script_dir):
            if f.startswith("gpu_test_") or f.startswith("ai_desktop_"):
                print(f"  - {f}")
    else:
        print(f"✗ Assessment failed with code {result.returncode}")
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
