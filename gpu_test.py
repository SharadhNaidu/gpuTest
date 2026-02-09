#!/usr/bin/env python3
"""
Comprehensive Desktop & GPU Testing Suite for AI Workloads
For College AI Lab - Desktop Review & Benchmarking
Generates detailed reports, analytics, and AI suitability analysis

Supports:
- Windows, Linux, macOS
- NVIDIA, AMD, Intel, Apple Silicon GPUs
- SSH/Headless/Cloud environments
- CPU, RAM, Storage, Network benchmarks
"""

import subprocess
import sys
import os
import json
import time
import datetime
import platform
import socket
import shutil
import tempfile
import hashlib
import struct
from pathlib import Path

# Detect OS
IS_WINDOWS = platform.system() == 'Windows'
IS_LINUX = platform.system() == 'Linux'
IS_MACOS = platform.system() == 'Darwin'
IS_ARM = platform.machine().lower() in ('arm64', 'aarch64', 'armv8')

def install_dependencies():
    """Install all required dependencies silently"""
    required_packages = ['numpy', 'matplotlib', 'pandas', 'psutil']
    missing = []
    
    for pkg in required_packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if not missing:
        return True
    
    print(f"[SETUP] Installing: {', '.join(missing)}...")
    
    # Try different installation methods
    methods = [
        [sys.executable, '-m', 'pip', 'install', '-q'] + missing,
        [sys.executable, '-m', 'pip', 'install', '-q', '--user'] + missing,
        [sys.executable, '-m', 'pip', 'install', '-q', '--break-system-packages'] + missing,
        [sys.executable, '-m', 'pip', 'install', '-q', '--user', '--break-system-packages'] + missing,
    ]
    
    for method in methods:
        try:
            subprocess.check_call(method, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Verify installation
            for pkg in missing:
                __import__(pkg)
            print("[SETUP] Dependencies installed successfully!")
            return True
        except:
            continue
    
    # If all methods failed, give instructions
    print(f"\n[ERROR] Could not auto-install packages. Please run manually:")
    print(f"    pip install {' '.join(missing)}")
    print(f"  or:")
    print(f"    pip install -r requirements.txt")
    print()
    sys.exit(1)

# Auto-install dependencies before importing
install_dependencies()

# Now import the packages
import psutil
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for SSH/headless
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Wedge, Rectangle
import pandas as pd

# Try to import GPU-specific libraries (optional)
CUDA_AVAILABLE = False
PYNVML_AVAILABLE = False
TORCH_AVAILABLE = False
TORCH_MPS_AVAILABLE = False  # Apple Silicon
TORCH_ROCM_AVAILABLE = False  # AMD ROCm
OPENCL_AVAILABLE = False

try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except:
    pass

try:
    import torch
    if torch.cuda.is_available():
        TORCH_AVAILABLE = True
        CUDA_AVAILABLE = True
    # Check for Apple Silicon MPS
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        TORCH_MPS_AVAILABLE = True
        TORCH_AVAILABLE = True
    # Check for AMD ROCm
    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
        TORCH_ROCM_AVAILABLE = True
        TORCH_AVAILABLE = True
except:
    pass

# Try OpenCL for AMD/Intel GPUs
try:
    import pyopencl as cl
    OPENCL_AVAILABLE = True
except:
    pass


class GPUTester:
    """Comprehensive Desktop & GPU Testing Class for AI Workloads"""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.datetime.now().isoformat(),
            'test_type': 'AI Desktop Assessment',
            'system_info': {},
            'cpu_info': {},
            'gpu_info': {},
            'memory_info': {},
            'storage_info': {},
            'network_info': {},
            'software_stack': {},
            'performance_tests': {},
            'stress_tests': {},
            'health_checks': {},
            'suitability_analysis': {},
            'recommendations': []
        }
        self.test_start_time = time.time()
        self.gpu_vendor = self._detect_gpu_vendor()
        
    def _detect_gpu_vendor(self):
        """Detect GPU vendor (NVIDIA, AMD, Intel, Apple, None)"""
        # Check NVIDIA
        if self._check_nvidia():
            return 'NVIDIA'
        # Check AMD
        if self._check_amd():
            return 'AMD'
        # Check Intel
        if self._check_intel():
            return 'Intel'
        # Check Apple Silicon
        if IS_MACOS and IS_ARM:
            return 'Apple'
        return 'None'
    
    def _check_nvidia(self):
        """Check if NVIDIA GPU is present"""
        try:
            if IS_WINDOWS:
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10,
                                       creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0)
            else:
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except:
            return False
    
    def _check_amd(self):
        """Check if AMD GPU is present"""
        try:
            if IS_LINUX:
                # Check for AMD ROCm
                result = subprocess.run(['rocm-smi'], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    return True
                # Check lspci
                result = subprocess.run(['lspci'], capture_output=True, text=True, timeout=10)
                if 'AMD' in result.stdout and ('VGA' in result.stdout or 'Display' in result.stdout):
                    return True
            elif IS_WINDOWS:
                # Check WMI for AMD GPU
                result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'],
                                       capture_output=True, text=True, timeout=10,
                                       creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0)
                if 'AMD' in result.stdout or 'Radeon' in result.stdout:
                    return True
        except:
            pass
        return False
    
    def _check_intel(self):
        """Check if Intel GPU is present"""
        try:
            if IS_LINUX:
                result = subprocess.run(['lspci'], capture_output=True, text=True, timeout=10)
                if 'Intel' in result.stdout and ('VGA' in result.stdout or 'Display' in result.stdout):
                    return True
            elif IS_WINDOWS:
                result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'],
                                       capture_output=True, text=True, timeout=10,
                                       creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0)
                if 'Intel' in result.stdout:
                    return True
        except:
            pass
        return False
        
    def get_nvidia_smi_info(self):
        """Get GPU info using nvidia-smi"""
        try:
            cmd = ['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used,memory.free,temperature.gpu,power.draw,power.limit,utilization.gpu,utilization.memory,pcie.link.gen.current,pcie.link.width.current,clocks.current.graphics,clocks.current.memory,clocks.max.graphics,clocks.max.memory,fan.speed,pstate',
                   '--format=csv,noheader,nounits']
            if IS_WINDOWS:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                                       creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0)
            else:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                values = [v.strip() for v in result.stdout.strip().split(',')]
                keys = ['name', 'driver_version', 'memory_total_mb', 'memory_used_mb', 'memory_free_mb',
                       'temperature_c', 'power_draw_w', 'power_limit_w', 'gpu_utilization_pct',
                       'memory_utilization_pct', 'pcie_gen', 'pcie_width', 'clock_graphics_mhz',
                       'clock_memory_mhz', 'clock_max_graphics_mhz', 'clock_max_memory_mhz', 
                       'fan_speed_pct', 'pstate']
                return dict(zip(keys, values))
        except Exception as e:
            return {'error': str(e)}
        return {}
    
    def get_amd_gpu_info(self):
        """Get AMD GPU info using rocm-smi or system tools"""
        info = {}
        try:
            if IS_LINUX:
                # Try rocm-smi first
                result = subprocess.run(['rocm-smi', '--showproductname', '--showtemp', '--showuse', 
                                        '--showmeminfo', 'vram', '--showpower'],
                                       capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    output = result.stdout
                    # Parse basic info
                    info['vendor'] = 'AMD'
                    if 'GPU' in output:
                        for line in output.split('\n'):
                            if 'Card series' in line or 'GPU' in line:
                                info['name'] = line.split(':')[-1].strip() if ':' in line else 'AMD GPU'
                            if 'Temperature' in line:
                                try:
                                    temp = ''.join(filter(str.isdigit, line.split(':')[-1]))
                                    info['temperature_c'] = temp
                                except:
                                    pass
                            if 'GPU use' in line:
                                try:
                                    info['gpu_utilization_pct'] = ''.join(filter(str.isdigit, line.split(':')[-1]))
                                except:
                                    pass
                    return info
            elif IS_WINDOWS:
                # Use WMI on Windows
                result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 
                                        'name,AdapterRAM,DriverVersion'],
                                       capture_output=True, text=True, timeout=30,
                                       creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0)
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'AMD' in line or 'Radeon' in line:
                            parts = line.split()
                            info['name'] = ' '.join([p for p in parts if not p.isdigit()])
                            info['vendor'] = 'AMD'
                            break
        except Exception as e:
            info['error'] = str(e)
        return info
    
    def get_intel_gpu_info(self):
        """Get Intel GPU info"""
        info = {}
        try:
            if IS_LINUX:
                result = subprocess.run(['lspci', '-v'], capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'Intel' in line and ('VGA' in line or 'Display' in line):
                            info['name'] = line.split(':')[-1].strip()
                            info['vendor'] = 'Intel'
                            break
            elif IS_WINDOWS:
                result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 
                                        'name,AdapterRAM,DriverVersion'],
                                       capture_output=True, text=True, timeout=30,
                                       creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0)
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'Intel' in line:
                            info['name'] = line.strip()
                            info['vendor'] = 'Intel'
                            break
        except Exception as e:
            info['error'] = str(e)
        return info
    
    def get_apple_gpu_info(self):
        """Get Apple Silicon GPU info"""
        info = {'vendor': 'Apple'}
        try:
            result = subprocess.run(['system_profiler', 'SPDisplaysDataType'],
                                   capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                output = result.stdout
                for line in output.split('\n'):
                    if 'Chipset Model' in line:
                        info['name'] = line.split(':')[-1].strip()
                    if 'VRAM' in line or 'Memory' in line:
                        info['memory'] = line.split(':')[-1].strip()
                    if 'Metal Support' in line:
                        info['metal_support'] = line.split(':')[-1].strip()
        except Exception as e:
            info['error'] = str(e)
        return info
    
    def get_cuda_version(self):
        """Get CUDA version"""
        try:
            if IS_WINDOWS:
                result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10,
                                       creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0)
            else:
                result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'release' in line.lower():
                        return line.split('release')[-1].split(',')[0].strip()
        except:
            pass
        
        try:
            if IS_WINDOWS:
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10,
                                       creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0)
            else:
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'CUDA Version' in line:
                        return line.split('CUDA Version:')[-1].split()[0].strip()
        except:
            pass
        return "Unknown"
    
    def collect_system_info(self):
        """Collect comprehensive system information"""
        print("[INFO] Collecting System Information...")
        
        # Get network hostname
        try:
            hostname = socket.gethostname()
            ip_address = socket.gethostbyname(hostname)
        except:
            hostname = platform.node()
            ip_address = "Unknown"
        
        # Detect if running via SSH or remotely
        is_remote = False
        if IS_LINUX or IS_MACOS:
            is_remote = 'SSH_CLIENT' in os.environ or 'SSH_TTY' in os.environ
        
        # Get CPU info
        cpu_info = self._get_cpu_detailed_info()
        
        self.results['system_info'] = {
            'hostname': hostname,
            'ip_address': ip_address,
            'os': platform.system(),
            'os_version': platform.release(),
            'os_full': f"{platform.system()} {platform.release()} {platform.version()}",
            'architecture': platform.machine(),
            'is_64bit': sys.maxsize > 2**32,
            'is_arm': IS_ARM,
            'python_version': platform.python_version(),
            'is_remote_session': is_remote,
            'environment': 'SSH/Remote' if is_remote else 'Local',
            **cpu_info
        }
        
        # Collect CPU-specific info
        self.results['cpu_info'] = cpu_info
    
    def _get_cpu_detailed_info(self):
        """Get detailed CPU information"""
        info = {
            'cpu_name': 'Unknown',
            'cpu_cores_physical': psutil.cpu_count(logical=False) or 0,
            'cpu_cores_logical': psutil.cpu_count(logical=True) or 0,
            'cpu_freq_mhz': 0,
            'cpu_freq_max_mhz': 0,
        }
        
        try:
            freq = psutil.cpu_freq()
            if freq:
                info['cpu_freq_mhz'] = round(freq.current, 0)
                info['cpu_freq_max_mhz'] = round(freq.max, 0) if freq.max else round(freq.current, 0)
        except:
            pass
        
        # Get CPU name
        try:
            if IS_LINUX:
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if 'model name' in line:
                            info['cpu_name'] = line.split(':')[1].strip()
                            break
            elif IS_WINDOWS:
                result = subprocess.run(['wmic', 'cpu', 'get', 'name'],
                                       capture_output=True, text=True, timeout=10,
                                       creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0)
                if result.returncode == 0:
                    lines = [l.strip() for l in result.stdout.split('\n') if l.strip() and 'Name' not in l]
                    if lines:
                        info['cpu_name'] = lines[0]
            elif IS_MACOS:
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                       capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    info['cpu_name'] = result.stdout.strip()
        except:
            pass
        
        # RAM info
        ram = psutil.virtual_memory()
        info['ram_total_gb'] = round(ram.total / (1024**3), 2)
        info['ram_available_gb'] = round(ram.available / (1024**3), 2)
        info['ram_used_pct'] = ram.percent
        
        return info
    
    def collect_storage_info(self):
        """Collect storage information and benchmark"""
        print("[INFO] Collecting Storage Information...")
        
        storage_info = {'disks': [], 'benchmark': {}}
        
        # Get disk partitions
        try:
            partitions = psutil.disk_partitions()
            for p in partitions:
                try:
                    usage = psutil.disk_usage(p.mountpoint)
                    storage_info['disks'].append({
                        'device': p.device,
                        'mountpoint': p.mountpoint,
                        'fstype': p.fstype,
                        'total_gb': round(usage.total / (1024**3), 2),
                        'used_gb': round(usage.used / (1024**3), 2),
                        'free_gb': round(usage.free / (1024**3), 2),
                        'used_pct': usage.percent
                    })
                except:
                    pass
        except:
            pass
        
        # Storage benchmark (simple sequential read/write test)
        storage_info['benchmark'] = self._run_storage_benchmark()
        
        self.results['storage_info'] = storage_info
    
    def _run_storage_benchmark(self):
        """Run a simple storage benchmark"""
        results = {}
        test_file = None
        try:
            # Create temp file
            test_dir = tempfile.gettempdir()
            test_file = os.path.join(test_dir, f'gpu_test_storage_{os.getpid()}.tmp')
            test_size_mb = 100
            block_size = 1024 * 1024  # 1MB blocks
            
            # Write test
            data = os.urandom(block_size)
            start = time.perf_counter()
            with open(test_file, 'wb') as f:
                for _ in range(test_size_mb):
                    f.write(data)
                f.flush()
                os.fsync(f.fileno())
            write_time = time.perf_counter() - start
            results['write_speed_mbps'] = round(test_size_mb / write_time, 2)
            
            # Read test
            start = time.perf_counter()
            with open(test_file, 'rb') as f:
                while f.read(block_size):
                    pass
            read_time = time.perf_counter() - start
            results['read_speed_mbps'] = round(test_size_mb / read_time, 2)
            
        except Exception as e:
            results['error'] = str(e)
        finally:
            if test_file and os.path.exists(test_file):
                try:
                    os.remove(test_file)
                except:
                    pass
        
        return results
    
    def collect_network_info(self):
        """Collect network information"""
        print("[INFO] Collecting Network Information...")
        
        network_info = {'interfaces': [], 'connectivity': {}}
        
        try:
            # Get network interfaces
            net_if = psutil.net_if_addrs()
            net_stats = psutil.net_if_stats()
            
            for iface, addrs in net_if.items():
                iface_info = {'name': iface, 'addresses': []}
                for addr in addrs:
                    if addr.family == socket.AF_INET:
                        iface_info['ipv4'] = addr.address
                    elif addr.family == socket.AF_INET6:
                        iface_info['ipv6'] = addr.address
                
                if iface in net_stats:
                    stats = net_stats[iface]
                    iface_info['speed_mbps'] = stats.speed
                    iface_info['is_up'] = stats.isup
                
                network_info['interfaces'].append(iface_info)
            
            # Test internet connectivity
            network_info['connectivity'] = self._test_network_connectivity()
            
        except Exception as e:
            network_info['error'] = str(e)
        
        self.results['network_info'] = network_info
    
    def _test_network_connectivity(self):
        """Test network connectivity"""
        results = {'internet': False, 'dns': False, 'latency_ms': None}
        
        try:
            # DNS test
            socket.gethostbyname('google.com')
            results['dns'] = True
            
            # Connectivity test with latency
            import urllib.request
            start = time.perf_counter()
            urllib.request.urlopen('http://www.google.com', timeout=5)
            latency = (time.perf_counter() - start) * 1000
            results['internet'] = True
            results['latency_ms'] = round(latency, 2)
        except:
            pass
        
        return results
    
    def collect_software_stack(self):
        """Collect information about installed AI/ML software"""
        print("[INFO] Analyzing Software Stack...")
        
        software = {
            'python_version': platform.python_version(),
            'packages': {},
            'frameworks': {}
        }
        
        # Check for common AI/ML packages
        packages_to_check = [
            'torch', 'tensorflow', 'keras', 'jax', 'flax',
            'numpy', 'scipy', 'pandas', 'scikit-learn',
            'opencv-python', 'pillow', 'transformers', 'diffusers',
            'onnx', 'onnxruntime', 'tensorrt', 'triton',
            'jupyter', 'jupyterlab', 'notebook',
            'cuda-python', 'cupy', 'numba'
        ]
        
        for pkg in packages_to_check:
            try:
                # Handle package name variations
                import_name = pkg.replace('-', '_').replace('opencv_python', 'cv2')
                mod = __import__(import_name.split('.')[0])
                version = getattr(mod, '__version__', 'installed')
                software['packages'][pkg] = version
            except:
                software['packages'][pkg] = 'not installed'
        
        # Check deep learning frameworks in detail
        if TORCH_AVAILABLE:
            software['frameworks']['pytorch'] = {
                'version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda if hasattr(torch.version, 'cuda') and torch.version.cuda else None,
                'cudnn_version': str(torch.backends.cudnn.version()) if torch.cuda.is_available() else None,
                'mps_available': TORCH_MPS_AVAILABLE,
                'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        
        try:
            import tensorflow as tf
            software['frameworks']['tensorflow'] = {
                'version': tf.__version__,
                'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0,
                'gpu_count': len(tf.config.list_physical_devices('GPU'))
            }
        except:
            pass
        
        self.results['software_stack'] = software
        
    def collect_gpu_info(self):
        """Collect GPU information based on vendor"""
        print(f"[INFO] Collecting GPU Information (Detected: {self.gpu_vendor})...")
        
        gpu_info = {
            'vendor': self.gpu_vendor,
            'cuda_available': CUDA_AVAILABLE,
            'torch_available': TORCH_AVAILABLE,
            'torch_mps_available': TORCH_MPS_AVAILABLE,
            'torch_rocm_available': TORCH_ROCM_AVAILABLE,
            'opencl_available': OPENCL_AVAILABLE,
            'pynvml_available': PYNVML_AVAILABLE,
        }
        
        if self.gpu_vendor == 'NVIDIA':
            nvidia_info = self.get_nvidia_smi_info()
            cuda_version = self.get_cuda_version()
            gpu_info.update(nvidia_info)
            gpu_info['cuda_version'] = cuda_version
            
            if TORCH_AVAILABLE and torch.cuda.is_available():
                gpu_info['torch_cuda_version'] = torch.version.cuda
                gpu_info['cudnn_version'] = str(torch.backends.cudnn.version())
                gpu_info['device_count'] = torch.cuda.device_count()
                gpu_info['current_device'] = torch.cuda.current_device()
                gpu_info['device_name'] = torch.cuda.get_device_name(0)
                gpu_info['device_capability'] = '.'.join(map(str, torch.cuda.get_device_capability(0)))
                
        elif self.gpu_vendor == 'AMD':
            amd_info = self.get_amd_gpu_info()
            gpu_info.update(amd_info)
            if TORCH_ROCM_AVAILABLE:
                gpu_info['rocm_version'] = torch.version.hip
                gpu_info['device_count'] = torch.cuda.device_count()
                gpu_info['device_name'] = torch.cuda.get_device_name(0)
                
        elif self.gpu_vendor == 'Intel':
            intel_info = self.get_intel_gpu_info()
            gpu_info.update(intel_info)
            
        elif self.gpu_vendor == 'Apple':
            apple_info = self.get_apple_gpu_info()
            gpu_info.update(apple_info)
            if TORCH_MPS_AVAILABLE:
                gpu_info['mps_device'] = 'mps'
                gpu_info['device_name'] = apple_info.get('name', 'Apple Silicon GPU')
        
        else:
            gpu_info['status'] = 'No discrete GPU detected'
            gpu_info['note'] = 'CPU-only mode - some benchmarks will be skipped'
        
        self.results['gpu_info'] = gpu_info
            
    def collect_memory_info(self):
        """Collect detailed memory information"""
        print("[INFO] Collecting Memory Information...")
        
        # System RAM info
        ram = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        self.results['memory_info'] = {
            'system_ram_total_gb': round(ram.total / (1024**3), 2),
            'system_ram_available_gb': round(ram.available / (1024**3), 2),
            'system_ram_used_pct': ram.percent,
            'swap_total_gb': round(swap.total / (1024**3), 2),
            'swap_used_gb': round(swap.used / (1024**3), 2),
        }
        
        # GPU VRAM info
        if TORCH_AVAILABLE and (torch.cuda.is_available() or TORCH_MPS_AVAILABLE):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.results['memory_info']['gpu_vram_total_gb'] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
                self.results['memory_info']['gpu_vram_allocated_gb'] = round(torch.cuda.memory_allocated(0) / (1024**3), 4)
                self.results['memory_info']['gpu_vram_cached_gb'] = round(torch.cuda.memory_reserved(0) / (1024**3), 4)
            elif TORCH_MPS_AVAILABLE:
                # Apple Silicon shares memory with system
                self.results['memory_info']['gpu_vram_total_gb'] = self.results['memory_info']['system_ram_total_gb']
                self.results['memory_info']['gpu_note'] = 'Unified Memory (shared with system)'
        elif 'memory_total_mb' in self.results['gpu_info']:
            try:
                self.results['memory_info']['gpu_vram_total_gb'] = round(float(self.results['gpu_info']['memory_total_mb']) / 1024, 2)
                self.results['memory_info']['gpu_vram_used_gb'] = round(float(self.results['gpu_info']['memory_used_mb']) / 1024, 2)
                self.results['memory_info']['gpu_vram_free_gb'] = round(float(self.results['gpu_info']['memory_free_mb']) / 1024, 2)
            except:
                pass
    
    def run_cpu_benchmark(self):
        """Run CPU performance benchmark"""
        print("  [TEST] Running CPU Benchmark...")
        
        results = {}
        
        # Single-threaded benchmark (matrix operations)
        try:
            size = 1000
            a = np.random.randn(size, size).astype(np.float32)
            b = np.random.randn(size, size).astype(np.float32)
            
            # Warmup
            _ = np.dot(a, b)
            
            # Benchmark
            iterations = 5
            start = time.perf_counter()
            for _ in range(iterations):
                _ = np.dot(a, b)
            elapsed = time.perf_counter() - start
            
            flops = 2 * (size ** 3) * iterations
            gflops = (flops / elapsed) / 1e9
            results['single_thread_gflops'] = round(gflops, 2)
            results['matmul_time_ms'] = round((elapsed / iterations) * 1000, 2)
        except Exception as e:
            results['single_thread_error'] = str(e)
        
        # Multi-threaded benchmark
        try:
            size = 2000
            a = np.random.randn(size, size).astype(np.float32)
            b = np.random.randn(size, size).astype(np.float32)
            
            start = time.perf_counter()
            _ = np.dot(a, b)  # NumPy uses all cores by default
            elapsed = time.perf_counter() - start
            
            flops = 2 * (size ** 3)
            gflops = (flops / elapsed) / 1e9
            results['multi_thread_gflops'] = round(gflops, 2)
        except Exception as e:
            results['multi_thread_error'] = str(e)
        
        return results
                
    def run_memory_bandwidth_test(self):
        """Test memory bandwidth"""
        print("  [TEST] Running Memory Bandwidth Test...")
        
        # First check if we have GPU support
        device = None
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                device = 'cuda'
            elif TORCH_MPS_AVAILABLE:
                device = 'mps'
        
        if device is None:
            # CPU-only memory bandwidth test
            return self._run_cpu_memory_bandwidth_test()
        
        results = {}
        sizes = [1, 10, 100, 500]  # MB
        
        for size_mb in sizes:
            try:
                n_elements = (size_mb * 1024 * 1024) // 4  # float32 = 4 bytes
                
                # Host to Device
                cpu_tensor = torch.randn(n_elements, dtype=torch.float32)
                if device == 'cuda':
                    torch.cuda.synchronize()
                start = time.perf_counter()
                gpu_tensor = cpu_tensor.to(device)
                if device == 'cuda':
                    torch.cuda.synchronize()
                elif device == 'mps':
                    torch.mps.synchronize()
                h2d_time = time.perf_counter() - start
                h2d_bandwidth = (size_mb / h2d_time) / 1024  # GB/s
                
                # Device to Host
                if device == 'cuda':
                    torch.cuda.synchronize()
                elif device == 'mps':
                    torch.mps.synchronize()
                start = time.perf_counter()
                cpu_back = gpu_tensor.cpu()
                if device == 'cuda':
                    torch.cuda.synchronize()
                elif device == 'mps':
                    torch.mps.synchronize()
                d2h_time = time.perf_counter() - start
                d2h_bandwidth = (size_mb / d2h_time) / 1024  # GB/s
                
                # Device to Device
                if device == 'cuda':
                    torch.cuda.synchronize()
                elif device == 'mps':
                    torch.mps.synchronize()
                start = time.perf_counter()
                gpu_tensor2 = gpu_tensor.clone()
                if device == 'cuda':
                    torch.cuda.synchronize()
                elif device == 'mps':
                    torch.mps.synchronize()
                d2d_time = time.perf_counter() - start
                d2d_bandwidth = (size_mb / d2d_time) / 1024  # GB/s
                
                results[f'{size_mb}MB'] = {
                    'h2d_bandwidth_gbps': round(h2d_bandwidth, 2),
                    'd2h_bandwidth_gbps': round(d2h_bandwidth, 2),
                    'd2d_bandwidth_gbps': round(d2d_bandwidth, 2)
                }
                
                del cpu_tensor, gpu_tensor, gpu_tensor2, cpu_back
                if device == 'cuda':
                    torch.cuda.empty_cache()
                
            except Exception as e:
                results[f'{size_mb}MB'] = {'error': str(e)}
                
        return results
    
    def _run_cpu_memory_bandwidth_test(self):
        """CPU-only memory bandwidth test using numpy"""
        results = {}
        sizes = [10, 100, 500]  # MB
        
        for size_mb in sizes:
            try:
                n_elements = (size_mb * 1024 * 1024) // 4
                
                # Allocate and copy test
                start = time.perf_counter()
                a = np.random.randn(n_elements).astype(np.float32)
                alloc_time = time.perf_counter() - start
                
                start = time.perf_counter()
                b = a.copy()
                copy_time = time.perf_counter() - start
                
                results[f'{size_mb}MB'] = {
                    'alloc_bandwidth_gbps': round((size_mb / alloc_time) / 1024, 2),
                    'copy_bandwidth_gbps': round((size_mb / copy_time) / 1024, 2),
                    'note': 'CPU-only test'
                }
                
                del a, b
                
            except Exception as e:
                results[f'{size_mb}MB'] = {'error': str(e)}
        
        return results
    
    def run_compute_benchmark(self):
        """Run compute performance benchmark"""
        print("  [TEST] Running Compute Benchmark...")
        
        # Determine device
        device = None
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                device = 'cuda'
            elif TORCH_MPS_AVAILABLE:
                device = 'mps'
        
        if device is None:
            return {'note': 'No GPU available - see CPU benchmark for compute performance'}
        
        results = {}
        
        # Matrix multiplication benchmark
        sizes = [1024, 2048, 4096]
        for size in sizes:
            try:
                a = torch.randn(size, size, device=device, dtype=torch.float32)
                b = torch.randn(size, size, device=device, dtype=torch.float32)
                
                # Warmup
                for _ in range(3):
                    c = torch.matmul(a, b)
                if device == 'cuda':
                    torch.cuda.synchronize()
                elif device == 'mps':
                    torch.mps.synchronize()
                
                # Benchmark
                iterations = 10
                start = time.perf_counter()
                for _ in range(iterations):
                    c = torch.matmul(a, b)
                if device == 'cuda':
                    torch.cuda.synchronize()
                elif device == 'mps':
                    torch.mps.synchronize()
                elapsed = time.perf_counter() - start
                
                # Calculate TFLOPS (2 * N^3 operations for matmul)
                flops = 2 * (size ** 3) * iterations
                tflops = (flops / elapsed) / 1e12
                
                results[f'matmul_{size}x{size}'] = {
                    'time_ms': round((elapsed / iterations) * 1000, 2),
                    'tflops': round(tflops, 2)
                }
                
                del a, b, c
                if device == 'cuda':
                    torch.cuda.empty_cache()
                
            except Exception as e:
                results[f'matmul_{size}x{size}'] = {'error': str(e)}
        
        # Convolution benchmark
        try:
            batch_sizes = [1, 8, 32]
            for batch in batch_sizes:
                x = torch.randn(batch, 64, 224, 224, device=device, dtype=torch.float32)
                conv = torch.nn.Conv2d(64, 128, 3, padding=1).to(device)
                
                # Warmup
                for _ in range(3):
                    y = conv(x)
                if device == 'cuda':
                    torch.cuda.synchronize()
                elif device == 'mps':
                    torch.mps.synchronize()
                
                # Benchmark
                iterations = 20
                start = time.perf_counter()
                for _ in range(iterations):
                    y = conv(x)
                if device == 'cuda':
                    torch.cuda.synchronize()
                elif device == 'mps':
                    torch.mps.synchronize()
                elapsed = time.perf_counter() - start
                
                results[f'conv2d_batch{batch}'] = {
                    'time_ms': round((elapsed / iterations) * 1000, 2),
                    'throughput_imgs_per_sec': round(batch * iterations / elapsed, 2)
                }
                
                del x, y, conv
                if device == 'cuda':
                    torch.cuda.empty_cache()
                
        except Exception as e:
            results['conv2d'] = {'error': str(e)}
            
        return results
    
    def run_stress_test(self, duration_seconds=30):
        """Run GPU stress test"""
        print(f"  [TEST] Running Stress Test ({duration_seconds}s)...")
        
        # Determine device
        device = None
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                device = 'cuda'
            elif TORCH_MPS_AVAILABLE:
                device = 'mps'
        
        if device is None:
            return self._run_cpu_stress_test(duration_seconds)
        
        results = {
            'device': device,
            'temperatures': [],
            'utilizations': [],
            'power_draws': [],
            'timestamps': []
        }
        
        try:
            # Create large tensors for stress
            size = 4096
            a = torch.randn(size, size, device=device, dtype=torch.float32)
            b = torch.randn(size, size, device=device, dtype=torch.float32)
            
            start_time = time.time()
            iteration = 0
            
            while time.time() - start_time < duration_seconds:
                # Perform heavy computation
                c = torch.matmul(a, b)
                c = torch.matmul(c, a)
                if device == 'cuda':
                    torch.cuda.synchronize()
                elif device == 'mps':
                    torch.mps.synchronize()
                
                # Collect metrics every second (NVIDIA only for detailed metrics)
                if iteration % 5 == 0:
                    results['timestamps'].append(round(time.time() - start_time, 1))
                    
                    if self.gpu_vendor == 'NVIDIA':
                        nvidia_info = self.get_nvidia_smi_info()
                        try:
                            results['temperatures'].append(float(nvidia_info.get('temperature_c', 0)))
                            results['utilizations'].append(float(nvidia_info.get('gpu_utilization_pct', 0)))
                            results['power_draws'].append(float(nvidia_info.get('power_draw_w', 0)))
                        except:
                            pass
                    else:
                        # For non-NVIDIA, just track CPU temp/usage as proxy
                        try:
                            results['utilizations'].append(psutil.cpu_percent())
                            # Try to get CPU temp on Linux
                            if IS_LINUX:
                                temps = psutil.sensors_temperatures()
                                if temps:
                                    for name, entries in temps.items():
                                        if entries:
                                            results['temperatures'].append(entries[0].current)
                                            break
                        except:
                            pass
                
                iteration += 1
            
            del a, b, c
            if device == 'cuda':
                torch.cuda.empty_cache()
            
            # Calculate statistics
            if results['temperatures']:
                results['max_temperature_c'] = max(results['temperatures'])
                results['avg_temperature_c'] = round(sum(results['temperatures']) / len(results['temperatures']), 1)
            if results['utilizations']:
                results['max_utilization_pct'] = max(results['utilizations'])
                results['avg_utilization_pct'] = round(sum(results['utilizations']) / len(results['utilizations']), 1)
            if results['power_draws']:
                results['max_power_w'] = max(results['power_draws'])
                results['avg_power_w'] = round(sum(results['power_draws']) / len(results['power_draws']), 1)
                    
        except Exception as e:
            results['error'] = str(e)
            
        return results
    
    def _run_cpu_stress_test(self, duration_seconds=30):
        """CPU-only stress test"""
        results = {
            'device': 'CPU',
            'temperatures': [],
            'utilizations': [],
            'timestamps': []
        }
        
        try:
            size = 2000
            start_time = time.time()
            iteration = 0
            
            while time.time() - start_time < duration_seconds:
                a = np.random.randn(size, size).astype(np.float32)
                b = np.random.randn(size, size).astype(np.float32)
                _ = np.dot(a, b)
                
                if iteration % 2 == 0:
                    results['timestamps'].append(round(time.time() - start_time, 1))
                    results['utilizations'].append(psutil.cpu_percent())
                    
                    # Try to get CPU temp on Linux
                    if IS_LINUX:
                        try:
                            temps = psutil.sensors_temperatures()
                            if temps:
                                for name, entries in temps.items():
                                    if entries:
                                        results['temperatures'].append(entries[0].current)
                                        break
                        except:
                            pass
                
                iteration += 1
            
            if results['temperatures']:
                results['max_temperature_c'] = max(results['temperatures'])
                results['avg_temperature_c'] = round(sum(results['temperatures']) / len(results['temperatures']), 1)
            if results['utilizations']:
                results['max_utilization_pct'] = max(results['utilizations'])
                results['avg_utilization_pct'] = round(sum(results['utilizations']) / len(results['utilizations']), 1)
                
        except Exception as e:
            results['error'] = str(e)
            
        return results
    
    def run_memory_stress_test(self):
        """Test maximum memory allocation"""
        print("  [TEST] Running Memory Stress Test...")
        
        # Determine device
        device = None
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                device = 'cuda'
            elif TORCH_MPS_AVAILABLE:
                device = 'mps'
        
        if device is None:
            return self._run_cpu_memory_stress_test()
        
        results = {'device': device}
        
        if device == 'cuda':
            torch.cuda.empty_cache()
            try:
                total_memory = torch.cuda.get_device_properties(0).total_memory
                allocated_tensors = []
                chunk_size = 256 * 1024 * 1024  # 256 MB chunks
                
                max_allocated = 0
                try:
                    while True:
                        t = torch.zeros(chunk_size // 4, device=device, dtype=torch.float32)
                        allocated_tensors.append(t)
                        max_allocated += chunk_size
                except RuntimeError:
                    pass
                
                results['max_allocatable_gb'] = round(max_allocated / (1024**3), 2)
                results['allocation_efficiency_pct'] = round((max_allocated / total_memory) * 100, 1)
                
                # Clean up
                del allocated_tensors
                torch.cuda.empty_cache()
                
                # Test memory fragmentation
                results['fragmentation_test'] = 'PASSED'
                try:
                    large_tensor = torch.zeros(int(total_memory * 0.8) // 4, device=device, dtype=torch.float32)
                    del large_tensor
                    torch.cuda.empty_cache()
                except:
                    results['fragmentation_test'] = 'FAILED'
                    
            except Exception as e:
                results['error'] = str(e)
        elif device == 'mps':
            # Apple Silicon - unified memory, test differently
            try:
                allocated_tensors = []
                chunk_size = 256 * 1024 * 1024
                max_allocated = 0
                
                # Only allocate up to 50% of system RAM to avoid system issues
                max_allowed = psutil.virtual_memory().total * 0.5
                
                try:
                    while max_allocated < max_allowed:
                        t = torch.zeros(chunk_size // 4, device=device, dtype=torch.float32)
                        allocated_tensors.append(t)
                        max_allocated += chunk_size
                except RuntimeError:
                    pass
                
                results['max_allocatable_gb'] = round(max_allocated / (1024**3), 2)
                results['note'] = 'Apple Silicon uses unified memory'
                
                del allocated_tensors
                
            except Exception as e:
                results['error'] = str(e)
                
        return results
    
    def _run_cpu_memory_stress_test(self):
        """CPU memory stress test"""
        results = {'device': 'CPU'}
        
        try:
            total_ram = psutil.virtual_memory().total
            allocated_arrays = []
            chunk_size = 256 * 1024 * 1024  # 256 MB
            max_allocated = 0
            
            # Only test up to 50% of RAM to avoid system issues
            max_allowed = total_ram * 0.5
            
            try:
                while max_allocated < max_allowed:
                    arr = np.zeros(chunk_size // 8, dtype=np.float64)
                    allocated_arrays.append(arr)
                    max_allocated += chunk_size
            except MemoryError:
                pass
            
            results['max_allocatable_gb'] = round(max_allocated / (1024**3), 2)
            results['allocation_pct_of_ram'] = round((max_allocated / total_ram) * 100, 1)
            
            del allocated_arrays
            
        except Exception as e:
            results['error'] = str(e)
            
        return results
    
    def run_error_check(self):
        """Check for GPU errors"""
        print("  [TEST] Running Error Checks...")
        
        results = {
            'cuda_errors': [],
            'memory_errors': 'NONE',
            'driver_errors': 'NONE',
            'system_health': 'OK'
        }
        
        # Check nvidia-smi for errors (NVIDIA only)
        if self.gpu_vendor == 'NVIDIA':
            try:
                if IS_WINDOWS:
                    result = subprocess.run(['nvidia-smi', '-q', '-d', 'ECC'],
                                           capture_output=True, text=True, timeout=30,
                                           creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0)
                else:
                    result = subprocess.run(['nvidia-smi', '-q', '-d', 'ECC'],
                                           capture_output=True, text=True, timeout=30)
                if 'error' in result.stdout.lower():
                    results['memory_errors'] = 'DETECTED'
            except:
                pass
        
        # Check system health
        try:
            # Check disk health
            disk_usage = psutil.disk_usage('/')
            if disk_usage.percent > 95:
                results['system_health'] = 'WARNING: Low disk space'
            
            # Check RAM
            ram = psutil.virtual_memory()
            if ram.percent > 95:
                results['system_health'] = 'WARNING: High memory usage'
        except:
            pass
        
        # Test basic GPU/tensor operations
        device = None
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                device = 'cuda'
            elif TORCH_MPS_AVAILABLE:
                device = 'mps'
        
        if device:
            try:
                # Test allocation
                t = torch.randn(1000, 1000, device=device)
                # Test computation
                r = torch.matmul(t, t)
                # Test synchronization
                if device == 'cuda':
                    torch.cuda.synchronize()
                elif device == 'mps':
                    torch.mps.synchronize()
                # Verify results
                assert not torch.isnan(r).any(), "NaN detected in computation"
                assert not torch.isinf(r).any(), "Inf detected in computation"
                results['basic_ops_test'] = 'PASSED'
                results['device_tested'] = device
                del t, r
                if device == 'cuda':
                    torch.cuda.empty_cache()
            except Exception as e:
                results['basic_ops_test'] = f'FAILED: {str(e)}'
                results['cuda_errors'].append(str(e))
        else:
            # CPU-only test
            try:
                t = np.random.randn(1000, 1000).astype(np.float32)
                r = np.dot(t, t)
                assert not np.isnan(r).any(), "NaN detected"
                assert not np.isinf(r).any(), "Inf detected"
                results['basic_ops_test'] = 'PASSED (CPU)'
                results['device_tested'] = 'CPU'
            except Exception as e:
                results['basic_ops_test'] = f'FAILED: {str(e)}'
        
        return results
    
    def run_performance_tests(self):
        """Run all performance tests"""
        print("\n[PHASE] Running Performance Tests...")
        
        self.results['performance_tests'] = {
            'cpu_benchmark': self.run_cpu_benchmark(),
            'memory_bandwidth': self.run_memory_bandwidth_test(),
            'compute_benchmark': self.run_compute_benchmark(),
        }
        
    def run_stress_tests(self):
        """Run all stress tests"""
        print("\n[PHASE] Running Stress Tests...")
        
        self.results['stress_tests'] = {
            'gpu_stress': self.run_stress_test(duration_seconds=30),
            'memory_stress': self.run_memory_stress_test(),
        }
        
    def run_health_checks(self):
        """Run all health checks"""
        print("\n[PHASE] Running Health Checks...")
        
        self.results['health_checks'] = self.run_error_check()
        
    def analyze_suitability(self):
        """Analyze system suitability for different AI workloads"""
        print("\n[PHASE] Analyzing AI Suitability...")
        
        scores = {
            'deep_learning_training': 0,
            'deep_learning_inference': 0,
            'scientific_computing': 0,
            'llm_inference': 0,
            'general_purpose': 0,
            'overall': 0
        }
        
        max_score = 100
        recommendations = []
        
        # ===== GPU/Accelerator Analysis =====
        gpu_vendor = self.results.get('gpu_info', {}).get('vendor', 'None')
        
        if gpu_vendor != 'None':
            # Has a GPU - good for AI
            scores['deep_learning_training'] += 15
            scores['deep_learning_inference'] += 15
            scores['llm_inference'] += 15
            
            # Memory analysis (GPU VRAM)
            try:
                mem_gb = float(self.results['memory_info'].get('gpu_vram_total_gb', 0))
                if mem_gb >= 24:
                    scores['deep_learning_training'] += 25
                    scores['deep_learning_inference'] += 20
                    scores['llm_inference'] += 25
                    scores['scientific_computing'] += 20
                elif mem_gb >= 16:
                    scores['deep_learning_training'] += 20
                    scores['deep_learning_inference'] += 20
                    scores['llm_inference'] += 20
                    scores['scientific_computing'] += 18
                elif mem_gb >= 8:
                    scores['deep_learning_training'] += 12
                    scores['deep_learning_inference'] += 18
                    scores['llm_inference'] += 12
                    scores['scientific_computing'] += 15
                    recommendations.append(f"[WARNING] Limited VRAM ({mem_gb}GB) may restrict large model training")
                elif mem_gb >= 4:
                    scores['deep_learning_training'] += 5
                    scores['deep_learning_inference'] += 12
                    scores['llm_inference'] += 5
                    scores['scientific_computing'] += 10
                    recommendations.append(f"[WARNING] Low VRAM ({mem_gb}GB) - suitable only for small models")
                else:
                    recommendations.append(f"[CRITICAL] Very low VRAM ({mem_gb}GB) - limited AI capability")
            except:
                recommendations.append("[WARNING] Could not analyze GPU memory capacity")
        else:
            recommendations.append("[INFO] No discrete GPU detected - CPU-only mode")
            scores['deep_learning_inference'] += 5  # Can still do CPU inference
        
        # ===== System RAM Analysis =====
        try:
            ram_gb = float(self.results['memory_info'].get('system_ram_total_gb', 0))
            if ram_gb >= 64:
                scores['deep_learning_training'] += 10
                scores['llm_inference'] += 15
                scores['scientific_computing'] += 15
                scores['general_purpose'] += 15
            elif ram_gb >= 32:
                scores['deep_learning_training'] += 8
                scores['llm_inference'] += 10
                scores['scientific_computing'] += 12
                scores['general_purpose'] += 12
            elif ram_gb >= 16:
                scores['deep_learning_training'] += 5
                scores['llm_inference'] += 5
                scores['scientific_computing'] += 8
                scores['general_purpose'] += 10
            else:
                recommendations.append(f"[WARNING] Limited RAM ({ram_gb}GB) - consider upgrading for AI workloads")
                scores['general_purpose'] += 5
        except:
            pass
        
        # ===== CPU Analysis =====
        try:
            cpu_cores = self.results.get('cpu_info', {}).get('cpu_cores_physical', 0)
            cpu_threads = self.results.get('cpu_info', {}).get('cpu_cores_logical', 0)
            
            if cpu_cores >= 16:
                scores['scientific_computing'] += 15
                scores['general_purpose'] += 10
                scores['deep_learning_training'] += 5  # For data loading
            elif cpu_cores >= 8:
                scores['scientific_computing'] += 10
                scores['general_purpose'] += 8
            elif cpu_cores >= 4:
                scores['scientific_computing'] += 5
                scores['general_purpose'] += 5
            
            # Check CPU benchmark results
            cpu_bench = self.results.get('performance_tests', {}).get('cpu_benchmark', {})
            if 'multi_thread_gflops' in cpu_bench:
                gflops = cpu_bench['multi_thread_gflops']
                if gflops >= 100:
                    scores['scientific_computing'] += 10
                elif gflops >= 50:
                    scores['scientific_computing'] += 5
        except:
            pass
        
        # ===== GPU Compute Performance Analysis =====
        try:
            compute = self.results['performance_tests'].get('compute_benchmark', {})
            matmul_4096 = compute.get('matmul_4096x4096', {})
            tflops = matmul_4096.get('tflops', 0)
            
            if tflops >= 15:
                scores['deep_learning_training'] += 20
                scores['deep_learning_inference'] += 15
                scores['scientific_computing'] += 15
            elif tflops >= 8:
                scores['deep_learning_training'] += 15
                scores['deep_learning_inference'] += 12
                scores['scientific_computing'] += 12
            elif tflops >= 4:
                scores['deep_learning_training'] += 10
                scores['deep_learning_inference'] += 10
                scores['scientific_computing'] += 8
            elif tflops > 0:
                scores['deep_learning_training'] += 5
                scores['deep_learning_inference'] += 8
                recommendations.append(f"[WARNING] Compute performance ({tflops} TFLOPS) is below optimal")
        except:
            pass
        
        # ===== Storage Analysis =====
        try:
            storage = self.results.get('storage_info', {}).get('benchmark', {})
            read_speed = storage.get('read_speed_mbps', 0)
            write_speed = storage.get('write_speed_mbps', 0)
            
            if read_speed >= 500 and write_speed >= 400:
                scores['deep_learning_training'] += 5
                scores['general_purpose'] += 5
            elif read_speed < 100 or write_speed < 100:
                recommendations.append("[WARNING] Slow storage may bottleneck data loading")
        except:
            pass
        
        # ===== Temperature Analysis =====
        try:
            stress = self.results['stress_tests'].get('gpu_stress', {})
            max_temp = stress.get('max_temperature_c', 0)
            
            if max_temp > 0:
                if max_temp < 75:
                    scores['deep_learning_training'] += 10
                    scores['deep_learning_inference'] += 10
                    scores['scientific_computing'] += 10
                elif max_temp < 85:
                    scores['deep_learning_training'] += 5
                    scores['deep_learning_inference'] += 8
                    scores['scientific_computing'] += 5
                    recommendations.append(f"[WARNING] Temperature under load ({max_temp}°C) - ensure adequate cooling")
                else:
                    recommendations.append(f"[CRITICAL] High temperature ({max_temp}°C) - thermal throttling likely")
        except:
            pass
        
        # ===== Health Check Analysis =====
        try:
            health = self.results['health_checks']
            if health.get('basic_ops_test', '').startswith('PASSED') and health.get('memory_errors') == 'NONE':
                scores['deep_learning_training'] += 10
                scores['deep_learning_inference'] += 15
                scores['scientific_computing'] += 10
                scores['general_purpose'] += 20
            else:
                recommendations.append("[CRITICAL] Health check failures detected - investigate errors")
        except:
            pass
        
        # ===== Software Stack Analysis =====
        try:
            software = self.results.get('software_stack', {})
            frameworks = software.get('frameworks', {})
            
            if 'pytorch' in frameworks:
                pytorch_info = frameworks['pytorch']
                if pytorch_info.get('cuda_available') or pytorch_info.get('mps_available'):
                    scores['deep_learning_training'] += 5
                    scores['deep_learning_inference'] += 5
            
            if 'tensorflow' in frameworks:
                tf_info = frameworks['tensorflow']
                if tf_info.get('gpu_available'):
                    scores['deep_learning_training'] += 3
                    scores['deep_learning_inference'] += 3
        except:
            pass
        
        # ===== Calculate Final Scores =====
        # Add base scores for working system
        scores['general_purpose'] = min(scores['general_purpose'] + 40, 100)
        
        # Ensure scores don't exceed 100
        for key in scores:
            scores[key] = min(scores[key], 100)
        
        # Calculate overall score
        scores['overall'] = round(sum([
            scores['deep_learning_training'],
            scores['deep_learning_inference'],
            scores['llm_inference'],
            scores['scientific_computing'],
            scores['general_purpose']
        ]) / 5)
        
        # ===== Generate Suitability Ratings =====
        suitability = {}
        for key, score in scores.items():
            if score >= 80:
                suitability[key] = {'score': score, 'rating': 'EXCELLENT', 'suitable': True}
            elif score >= 60:
                suitability[key] = {'score': score, 'rating': 'GOOD', 'suitable': True}
            elif score >= 40:
                suitability[key] = {'score': score, 'rating': 'MODERATE', 'suitable': True}
            elif score >= 20:
                suitability[key] = {'score': score, 'rating': 'LIMITED', 'suitable': False}
            else:
                suitability[key] = {'score': score, 'rating': 'POOR', 'suitable': False}
        
        # ===== Add Recommendations =====
        if not recommendations:
            recommendations.append("[OK] System is well-configured for AI workloads")
        
        # Add specific recommendations based on scores
        if scores['deep_learning_training'] < 40:
            recommendations.append("[SUGGESTION] Consider a dedicated GPU for deep learning training")
        if scores['llm_inference'] >= 60:
            recommendations.append("[OK] System suitable for running local LLMs (7B-13B models)")
        if scores['llm_inference'] >= 80:
            recommendations.append("[OK] System suitable for larger LLMs (30B+ models)")
        
        self.results['suitability_analysis'] = suitability
        self.results['recommendations'] = recommendations
        
    def generate_report_image(self, output_path='gpu_report.png'):
        """Generate comprehensive report image"""
        print("\n[INFO] Generating Report Image...")
        
        fig = plt.figure(figsize=(24, 30))
        fig.suptitle('AI Desktop Testing & Analysis Report\nCollege AI Lab Assessment', fontsize=22, fontweight='bold', y=0.98)
        
        # Create grid
        gs = gridspec.GridSpec(8, 4, figure=fig, hspace=0.4, wspace=0.3)
        
        # Colors
        colors = {
            'excellent': '#2ecc71',
            'good': '#3498db',
            'moderate': '#f39c12',
            'poor': '#e74c3c',
            'background': '#2c3e50',
            'text': '#ecf0f1'
        }
        
        # 1. System & Hardware Info Panel
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.set_facecolor('#34495e')
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        ax1.axis('off')
        ax1.set_title('System Information', fontsize=14, fontweight='bold', pad=10)
        
        sys_info = self.results.get('system_info', {})
        cpu_info = self.results.get('cpu_info', {})
        
        info_text = f"""
Hostname: {sys_info.get('hostname', 'N/A')}
OS: {sys_info.get('os_full', sys_info.get('os', 'N/A'))}
Architecture: {sys_info.get('architecture', 'N/A')} {'(ARM)' if sys_info.get('is_arm') else ''}
Environment: {sys_info.get('environment', 'Local')}

CPU: {cpu_info.get('cpu_name', 'N/A')}
Cores: {cpu_info.get('cpu_cores_physical', 'N/A')} physical / {cpu_info.get('cpu_cores_logical', 'N/A')} logical
Frequency: {cpu_info.get('cpu_freq_mhz', 'N/A')} MHz (Max: {cpu_info.get('cpu_freq_max_mhz', 'N/A')} MHz)
RAM: {cpu_info.get('ram_total_gb', 'N/A')} GB
        """
        ax1.text(0.5, 5, info_text, fontsize=10, verticalalignment='center', 
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='#2c3e50', alpha=0.8))
        
        # 2. GPU Info Panel
        ax2 = fig.add_subplot(gs[0, 2:])
        ax2.set_facecolor('#34495e')
        ax2.axis('off')
        ax2.set_title('GPU/Accelerator Information', fontsize=14, fontweight='bold', pad=10)
        
        gpu_info = self.results.get('gpu_info', {})
        mem_info = self.results.get('memory_info', {})
        
        gpu_name = gpu_info.get('name', gpu_info.get('device_name', 'No GPU detected'))
        gpu_vendor = gpu_info.get('vendor', 'Unknown')
        
        gpu_text = f"""
Vendor: {gpu_vendor}
GPU: {gpu_name}
Driver: {gpu_info.get('driver_version', 'N/A')}
CUDA/ROCm: {gpu_info.get('cuda_version', gpu_info.get('rocm_version', 'N/A'))}
Compute: {gpu_info.get('device_capability', 'N/A')}

VRAM: {mem_info.get('gpu_vram_total_gb', 'N/A')} GB
System RAM: {mem_info.get('system_ram_total_gb', 'N/A')} GB
PyTorch: {'Available' if gpu_info.get('torch_available') else 'Not Available'}
        """
        ax2.text(0.5, 0.5, gpu_text, fontsize=10, verticalalignment='center',
                transform=ax2.transAxes, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#2c3e50', alpha=0.8))
        
        # 3. CPU Benchmark Chart
        ax3 = fig.add_subplot(gs[1, :2])
        cpu_bench = self.results.get('performance_tests', {}).get('cpu_benchmark', {})
        
        if cpu_bench and 'single_thread_gflops' in cpu_bench:
            categories = ['Single-Thread', 'Multi-Thread']
            values = [
                cpu_bench.get('single_thread_gflops', 0),
                cpu_bench.get('multi_thread_gflops', 0)
            ]
            bars = ax3.bar(categories, values, color=['#3498db', '#2ecc71'])
            ax3.set_ylabel('GFLOPS', fontsize=12)
            ax3.set_title('CPU Performance (Matrix Multiplication)', fontsize=14, fontweight='bold')
            for bar, val in zip(bars, values):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val}', ha='center', fontsize=10, fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'No CPU benchmark data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('CPU Performance', fontsize=14, fontweight='bold')
        
        # 4. Storage Performance
        ax4 = fig.add_subplot(gs[1, 2:])
        storage = self.results.get('storage_info', {}).get('benchmark', {})
        
        if storage and 'read_speed_mbps' in storage:
            categories = ['Read Speed', 'Write Speed']
            values = [storage.get('read_speed_mbps', 0), storage.get('write_speed_mbps', 0)]
            colors_storage = ['#2ecc71', '#e74c3c']
            bars = ax4.bar(categories, values, color=colors_storage)
            ax4.set_ylabel('MB/s', fontsize=12)
            ax4.set_title('Storage Performance', fontsize=14, fontweight='bold')
            for bar, val in zip(bars, values):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        f'{val} MB/s', ha='center', fontsize=10, fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'No storage benchmark data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Storage Performance', fontsize=14, fontweight='bold')
        
        # 5. GPU Compute Performance Chart
        ax5 = fig.add_subplot(gs[2, :2])
        compute = self.results.get('performance_tests', {}).get('compute_benchmark', {})
        
        matmul_sizes = []
        matmul_tflops = []
        for key in ['matmul_1024x1024', 'matmul_2048x2048', 'matmul_4096x4096']:
            if key in compute and 'tflops' in compute[key]:
                matmul_sizes.append(key.replace('matmul_', ''))
                matmul_tflops.append(compute[key]['tflops'])
        
        if matmul_tflops:
            bars = ax5.bar(matmul_sizes, matmul_tflops, color=['#3498db', '#2ecc71', '#e74c3c'])
            ax5.set_ylabel('TFLOPS', fontsize=12)
            ax5.set_title('GPU Matrix Multiplication Performance', fontsize=14, fontweight='bold')
            ax5.set_xlabel('Matrix Size')
            for bar, val in zip(bars, matmul_tflops):
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{val}', ha='center', fontsize=10, fontweight='bold')
        else:
            ax5.text(0.5, 0.5, 'No GPU compute data\n(No GPU or PyTorch unavailable)', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('GPU Compute Performance', fontsize=14, fontweight='bold')
        
        # 6. Convolution Performance
        ax6 = fig.add_subplot(gs[2, 2:])
        conv_batches = []
        conv_throughput = []
        for key in ['conv2d_batch1', 'conv2d_batch8', 'conv2d_batch32']:
            if key in compute and 'throughput_imgs_per_sec' in compute[key]:
                conv_batches.append(key.replace('conv2d_batch', 'Batch '))
                conv_throughput.append(compute[key]['throughput_imgs_per_sec'])
        
        if conv_throughput:
            bars = ax6.bar(conv_batches, conv_throughput, color=['#9b59b6', '#1abc9c', '#f39c12'])
            ax6.set_ylabel('Images/sec', fontsize=12)
            ax6.set_title('Conv2D Performance (224x224)', fontsize=14, fontweight='bold')
            for bar, val in zip(bars, conv_throughput):
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{val:.0f}', ha='center', fontsize=10, fontweight='bold')
        else:
            ax6.text(0.5, 0.5, 'No convolution data', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Conv2D Performance', fontsize=14, fontweight='bold')
        
        # 7. Stress Test - Temperature
        ax7 = fig.add_subplot(gs[3, :2])
        stress = self.results.get('stress_tests', {}).get('gpu_stress', {})
        temps = stress.get('temperatures', [])
        timestamps = stress.get('timestamps', [])
        
        if temps and timestamps and len(temps) == len(timestamps):
            ax7.plot(timestamps, temps, 'r-', linewidth=2, marker='o', markersize=4)
            ax7.fill_between(timestamps, temps, alpha=0.3, color='red')
            ax7.axhline(y=80, color='orange', linestyle='--', label='Warning (80°C)')
            ax7.axhline(y=90, color='red', linestyle='--', label='Critical (90°C)')
            ax7.set_xlabel('Time (s)')
            ax7.set_ylabel('Temperature (°C)')
            device_type = stress.get('device', 'GPU')
            max_temp = stress.get('max_temperature_c', 'N/A')
            ax7.set_title(f'{device_type} Temperature Under Stress (Max: {max_temp}°C)', 
                         fontsize=14, fontweight='bold')
            ax7.legend(loc='upper right')
            ax7.grid(True, alpha=0.3)
        else:
            ax7.text(0.5, 0.5, 'No temperature data\n(May not be available for this GPU type)', ha='center', va='center', transform=ax7.transAxes)
            ax7.set_title('Temperature Under Stress', fontsize=14, fontweight='bold')
        
        # 8. Stress Test - Utilization
        ax8 = fig.add_subplot(gs[3, 2:])
        utils = stress.get('utilizations', [])
        
        if utils and timestamps and len(utils) == len(timestamps):
            ax8.plot(timestamps, utils, 'b-', linewidth=2, label='Utilization %')
            ax8.fill_between(timestamps, utils, alpha=0.3, color='blue')
            ax8.set_ylabel('Utilization (%)', color='blue')
            ax8.set_xlabel('Time (s)')
            device_type = stress.get('device', 'GPU')
            ax8.set_title(f'{device_type} Utilization Under Stress', fontsize=14, fontweight='bold')
            ax8.grid(True, alpha=0.3)
        else:
            ax8.text(0.5, 0.5, 'No utilization data', ha='center', va='center', transform=ax8.transAxes)
            ax8.set_title('Utilization Under Stress', fontsize=14, fontweight='bold')
        
        # 9. Memory Bandwidth
        ax9 = fig.add_subplot(gs[4, :2])
        bw_test = self.results.get('performance_tests', {}).get('memory_bandwidth', {})
        
        if bw_test and '100MB' in bw_test:
            bw_data = bw_test['100MB']
            if 'h2d_bandwidth_gbps' in bw_data:
                categories = ['Host→Device', 'Device→Host', 'Device→Device']
                values = [
                    bw_data.get('h2d_bandwidth_gbps', 0),
                    bw_data.get('d2h_bandwidth_gbps', 0),
                    bw_data.get('d2d_bandwidth_gbps', 0)
                ]
            else:
                categories = ['Allocate', 'Copy']
                values = [
                    bw_data.get('alloc_bandwidth_gbps', 0),
                    bw_data.get('copy_bandwidth_gbps', 0)
                ]
            colors_bw = ['#3498db', '#e74c3c', '#2ecc71'][:len(categories)]
            bars = ax9.barh(categories, values, color=colors_bw)
            ax9.set_xlabel('Bandwidth (GB/s)')
            ax9.set_title('Memory Bandwidth (100MB Transfer)', fontsize=14, fontweight='bold')
            for bar, val in zip(bars, values):
                ax9.text(val + 0.1, bar.get_y() + bar.get_height()/2, 
                        f'{val} GB/s', va='center', fontsize=10, fontweight='bold')
        else:
            ax9.text(0.5, 0.5, 'No bandwidth data', ha='center', va='center', transform=ax9.transAxes)
            ax9.set_title('Memory Bandwidth', fontsize=14, fontweight='bold')
        
        # 10. Health Check Status
        ax10 = fig.add_subplot(gs[4, 2:])
        ax10.axis('off')
        ax10.set_title('Health Check Status', fontsize=14, fontweight='bold', pad=10)
        
        health = self.results.get('health_checks', {})
        mem_stress = self.results.get('stress_tests', {}).get('memory_stress', {})
        
        health_items = [
            ('Basic Operations', health.get('basic_ops_test', 'N/A')),
            ('Memory Errors', health.get('memory_errors', 'N/A')),
            ('System Health', health.get('system_health', 'N/A')),
            ('Memory Stress', f"{mem_stress.get('max_allocatable_gb', 'N/A')} GB allocatable")
        ]
        
        y_pos = 0.85
        for item, status in health_items:
            if isinstance(status, str):
                color = '#2ecc71' if any(x in status for x in ['PASSED', 'NONE', 'OK']) else '#f39c12' if 'WARNING' in status else '#e74c3c'
                symbol = '✓' if any(x in status for x in ['PASSED', 'NONE', 'OK']) else '⚠' if 'WARNING' in status else '✗'
            else:
                color = '#3498db'
                symbol = '•'
            ax10.text(0.1, y_pos, f'{symbol} {item}: {status}', fontsize=11, 
                    transform=ax10.transAxes, color=color, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            y_pos -= 0.2
        
        # 11. Software Stack
        ax11 = fig.add_subplot(gs[5, :2])
        ax11.axis('off')
        ax11.set_title('AI/ML Software Stack', fontsize=14, fontweight='bold', pad=10)
        
        software = self.results.get('software_stack', {})
        packages = software.get('packages', {})
        
        # Show key packages
        key_packages = ['torch', 'tensorflow', 'numpy', 'transformers', 'onnxruntime']
        sw_text = f"Python: {software.get('python_version', 'N/A')}\n\n"
        for pkg in key_packages:
            status = packages.get(pkg, 'not checked')
            icon = '✓' if status != 'not installed' else '✗'
            sw_text += f"{icon} {pkg}: {status}\n"
        
        ax11.text(0.5, 0.5, sw_text, fontsize=10, verticalalignment='center',
                 transform=ax11.transAxes, fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.9))
        
        # 12. Network Info
        ax12 = fig.add_subplot(gs[5, 2:])
        ax12.axis('off')
        ax12.set_title('Network Status', fontsize=14, fontweight='bold', pad=10)
        
        network = self.results.get('network_info', {})
        connectivity = network.get('connectivity', {})
        
        net_text = f"""
Internet: {'✓ Connected' if connectivity.get('internet') else '✗ No Connection'}
DNS: {'✓ Working' if connectivity.get('dns') else '✗ Failed'}
Latency: {connectivity.get('latency_ms', 'N/A')} ms

Interfaces: {len(network.get('interfaces', []))}
        """
        ax12.text(0.5, 0.5, net_text, fontsize=11, verticalalignment='center',
                 transform=ax12.transAxes, fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.9))
        
        # 13. Suitability Gauges
        ax13 = fig.add_subplot(gs[6, :])
        ax13.axis('off')
        ax13.set_title('AI Workload Suitability Analysis', fontsize=16, fontweight='bold', pad=20)
        
        suitability = self.results.get('suitability_analysis', {})
        categories = ['DL Training', 'DL Inference', 'LLM Inference', 'Scientific\nComputing', 'General\nPurpose', 'Overall']
        keys = ['deep_learning_training', 'deep_learning_inference', 'llm_inference', 'scientific_computing', 'general_purpose', 'overall']
        
        for i, (cat, key) in enumerate(zip(categories, keys)):
            data = suitability.get(key, {})
            score = data.get('score', 0)
            rating = data.get('rating', 'N/A')
            
            # Position for each gauge
            x_center = 0.08 + i * 0.155
            
            # Draw gauge background
            gauge_ax = fig.add_axes([x_center - 0.06, 0.18, 0.12, 0.08])
            gauge_ax.axis('off')
            
            # Color based on score
            if score >= 80:
                bar_color = '#2ecc71'
            elif score >= 60:
                bar_color = '#3498db'
            elif score >= 40:
                bar_color = '#f39c12'
            else:
                bar_color = '#e74c3c'
            
            # Draw progress bar
            gauge_ax.barh([0], [100], color='#ecf0f1', height=0.5)
            gauge_ax.barh([0], [score], color=bar_color, height=0.5)
            gauge_ax.set_xlim(0, 100)
            gauge_ax.set_ylim(-0.5, 0.5)
            
            # Add text
            gauge_ax.text(50, -0.9, cat, ha='center', va='top', fontsize=9, fontweight='bold')
            gauge_ax.text(50, 0, f'{score}%\n{rating}', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        
        # 14. Recommendations
        ax14 = fig.add_subplot(gs[7, :])
        ax14.axis('off')
        ax14.set_title('Recommendations & Findings', fontsize=14, fontweight='bold', pad=10)
        
        recommendations = self.results.get('recommendations', [])
        if not recommendations:
            recommendations = ['[OK] System is performing within expected parameters']
        
        rec_text = '\n'.join(recommendations[:8])  # Limit to 8 recommendations
        ax14.text(0.5, 0.5, rec_text, fontsize=11, ha='center', va='center',
                 transform=ax14.transAxes, fontfamily='sans-serif',
                 bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.9))
        
        # Add timestamp and footer
        test_duration = round(time.time() - self.test_start_time, 1)
        fig.text(0.5, 0.01, f'Report Generated: {self.results["timestamp"]} | Test Duration: {test_duration}s | GPU Vendor: {self.gpu_vendor}',
                ha='center', fontsize=10, style='italic')
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"  [OK] Report image saved: {output_path}")
        return output_path
    
    def generate_text_report(self, output_path='gpu_report.txt'):
        """Generate detailed text report"""
        print("[INFO] Generating Text Report...")
        
        report = []
        report.append("=" * 80)
        report.append("AI DESKTOP TESTING & ANALYSIS REPORT")
        report.append("College AI Lab - Comprehensive System Assessment")
        report.append("=" * 80)
        report.append(f"\nReport Generated: {self.results['timestamp']}")
        report.append(f"Test Duration: {round(time.time() - self.test_start_time, 1)} seconds")
        report.append(f"Test Type: {self.results.get('test_type', 'Desktop Assessment')}\n")
        
        # System Info
        report.append("\n" + "=" * 40)
        report.append("SYSTEM INFORMATION")
        report.append("=" * 40)
        sys_info = self.results.get('system_info', {})
        for key, value in sys_info.items():
            report.append(f"  {key.replace('_', ' ').title()}: {value}")
        
        # CPU Info
        report.append("\n" + "=" * 40)
        report.append("CPU INFORMATION")
        report.append("=" * 40)
        cpu_info = self.results.get('cpu_info', {})
        for key, value in cpu_info.items():
            report.append(f"  {key.replace('_', ' ').title()}: {value}")
        
        # GPU Info
        report.append("\n" + "=" * 40)
        report.append("GPU/ACCELERATOR INFORMATION")
        report.append("=" * 40)
        gpu_info = self.results.get('gpu_info', {})
        important_keys = ['vendor', 'name', 'device_name', 'driver_version', 'cuda_version', 'rocm_version',
                         'device_capability', 'memory_total_mb', 'temperature_c', 'power_limit_w', 
                         'pcie_gen', 'pcie_width', 'metal_support', 'torch_available', 'cuda_available']
        for key in important_keys:
            if key in gpu_info:
                report.append(f"  {key.replace('_', ' ').title()}: {gpu_info[key]}")
        
        # Memory Info
        report.append("\n" + "=" * 40)
        report.append("MEMORY INFORMATION")
        report.append("=" * 40)
        for key, value in self.results.get('memory_info', {}).items():
            report.append(f"  {key.replace('_', ' ').title()}: {value}")
        
        # Storage Info
        report.append("\n" + "=" * 40)
        report.append("STORAGE INFORMATION")
        report.append("=" * 40)
        storage = self.results.get('storage_info', {})
        for disk in storage.get('disks', [])[:3]:  # Show first 3 disks
            report.append(f"  {disk.get('mountpoint', 'N/A')}: {disk.get('total_gb', 'N/A')} GB total, {disk.get('free_gb', 'N/A')} GB free ({disk.get('used_pct', 'N/A')}% used)")
        benchmark = storage.get('benchmark', {})
        if benchmark:
            report.append(f"  Read Speed: {benchmark.get('read_speed_mbps', 'N/A')} MB/s")
            report.append(f"  Write Speed: {benchmark.get('write_speed_mbps', 'N/A')} MB/s")
        
        # Network Info
        report.append("\n" + "=" * 40)
        report.append("NETWORK INFORMATION")
        report.append("=" * 40)
        network = self.results.get('network_info', {})
        connectivity = network.get('connectivity', {})
        report.append(f"  Internet Connected: {connectivity.get('internet', 'N/A')}")
        report.append(f"  DNS Working: {connectivity.get('dns', 'N/A')}")
        report.append(f"  Latency: {connectivity.get('latency_ms', 'N/A')} ms")
        
        # Software Stack
        report.append("\n" + "=" * 40)
        report.append("AI/ML SOFTWARE STACK")
        report.append("=" * 40)
        software = self.results.get('software_stack', {})
        report.append(f"  Python Version: {software.get('python_version', 'N/A')}")
        packages = software.get('packages', {})
        for pkg, ver in packages.items():
            status = "✓" if ver != 'not installed' else "✗"
            report.append(f"  {status} {pkg}: {ver}")
        
        # Performance Tests
        report.append("\n" + "=" * 40)
        report.append("PERFORMANCE TEST RESULTS")
        report.append("=" * 40)
        
        # CPU Benchmark
        report.append("\n  CPU Benchmark:")
        cpu_bench = self.results.get('performance_tests', {}).get('cpu_benchmark', {})
        report.append(f"    Single-Thread: {cpu_bench.get('single_thread_gflops', 'N/A')} GFLOPS")
        report.append(f"    Multi-Thread: {cpu_bench.get('multi_thread_gflops', 'N/A')} GFLOPS")
        
        # Memory Bandwidth
        report.append("\n  Memory Bandwidth Test:")
        bw_test = self.results.get('performance_tests', {}).get('memory_bandwidth', {})
        for size, data in bw_test.items():
            if isinstance(data, dict) and 'error' not in data:
                report.append(f"    {size}:")
                if 'h2d_bandwidth_gbps' in data:
                    report.append(f"      Host to Device: {data.get('h2d_bandwidth_gbps', 'N/A')} GB/s")
                    report.append(f"      Device to Host: {data.get('d2h_bandwidth_gbps', 'N/A')} GB/s")
                    report.append(f"      Device to Device: {data.get('d2d_bandwidth_gbps', 'N/A')} GB/s")
                else:
                    report.append(f"      Copy Bandwidth: {data.get('copy_bandwidth_gbps', 'N/A')} GB/s")
        
        # GPU Compute Benchmark
        report.append("\n  GPU Compute Benchmark:")
        compute = self.results.get('performance_tests', {}).get('compute_benchmark', {})
        for test, data in compute.items():
            if isinstance(data, dict) and 'error' not in data:
                if 'tflops' in data:
                    report.append(f"    {test}: {data['tflops']} TFLOPS ({data.get('time_ms', 'N/A')} ms)")
                elif 'throughput_imgs_per_sec' in data:
                    report.append(f"    {test}: {data['throughput_imgs_per_sec']} images/sec ({data.get('time_ms', 'N/A')} ms)")
        
        # Stress Tests
        report.append("\n" + "=" * 40)
        report.append("STRESS TEST RESULTS")
        report.append("=" * 40)
        stress = self.results.get('stress_tests', {}).get('gpu_stress', {})
        report.append(f"  Test Device: {stress.get('device', 'N/A')}")
        report.append(f"  Maximum Temperature: {stress.get('max_temperature_c', 'N/A')}°C")
        report.append(f"  Average Temperature: {stress.get('avg_temperature_c', 'N/A')}°C")
        report.append(f"  Maximum Utilization: {stress.get('max_utilization_pct', 'N/A')}%")
        report.append(f"  Average Utilization: {stress.get('avg_utilization_pct', 'N/A')}%")
        report.append(f"  Maximum Power Draw: {stress.get('max_power_w', 'N/A')}W")
        report.append(f"  Average Power Draw: {stress.get('avg_power_w', 'N/A')}W")
        
        mem_stress = self.results.get('stress_tests', {}).get('memory_stress', {})
        report.append(f"\n  Memory Stress Test:")
        report.append(f"    Test Device: {mem_stress.get('device', 'N/A')}")
        report.append(f"    Max Allocatable: {mem_stress.get('max_allocatable_gb', 'N/A')} GB")
        report.append(f"    Allocation Efficiency: {mem_stress.get('allocation_efficiency_pct', 'N/A')}%")
        report.append(f"    Fragmentation Test: {mem_stress.get('fragmentation_test', 'N/A')}")
        
        # Health Checks
        report.append("\n" + "=" * 40)
        report.append("HEALTH CHECK RESULTS")
        report.append("=" * 40)
        health = self.results.get('health_checks', {})
        report.append(f"  Basic Operations Test: {health.get('basic_ops_test', 'N/A')}")
        report.append(f"  Device Tested: {health.get('device_tested', 'N/A')}")
        report.append(f"  Memory Errors: {health.get('memory_errors', 'N/A')}")
        report.append(f"  System Health: {health.get('system_health', 'N/A')}")
        
        # Suitability Analysis
        report.append("\n" + "=" * 40)
        report.append("AI WORKLOAD SUITABILITY ANALYSIS")
        report.append("=" * 40)
        suitability = self.results.get('suitability_analysis', {})
        for key, data in suitability.items():
            if isinstance(data, dict):
                suitable = "✓ SUITABLE" if data.get('suitable', False) else "✗ NOT SUITABLE"
                report.append(f"  {key.replace('_', ' ').title()}: {data.get('score', 0)}% ({data.get('rating', 'N/A')}) - {suitable}")
        
        # Recommendations
        report.append("\n" + "=" * 40)
        report.append("RECOMMENDATIONS")
        report.append("=" * 40)
        recommendations = self.results.get('recommendations', [])
        if recommendations:
            for rec in recommendations:
                report.append(f"  {rec}")
        else:
            report.append("  [OK] System is performing within expected parameters")
            report.append("  [OK] No issues detected - system is suitable for AI workloads")
        
        # Final Verdict
        report.append("\n" + "=" * 40)
        report.append("FINAL VERDICT")
        report.append("=" * 40)
        overall = suitability.get('overall', {})
        overall_score = overall.get('score', 0)
        
        if overall_score >= 70:
            verdict = "SYSTEM IS WELL-SUITED FOR AI WORKLOADS"
            verdict_detail = "The system passed all critical tests and is recommended for AI/ML tasks."
        elif overall_score >= 50:
            verdict = "SYSTEM IS CONDITIONALLY SUITABLE"
            verdict_detail = "The system can handle basic AI tasks but has some limitations. Review recommendations."
        elif overall_score >= 30:
            verdict = "SYSTEM HAS LIMITED AI CAPABILITY"
            verdict_detail = "The system can handle inference tasks but is not ideal for training. Consider upgrades."
        else:
            verdict = "SYSTEM REQUIRES UPGRADES FOR AI WORKLOADS"
            verdict_detail = "The system has significant limitations. Consider hardware upgrades for AI tasks."
        
        report.append(f"\n  >>> {verdict} <<<")
        report.append(f"  Overall Score: {overall_score}%")
        report.append(f"  {verdict_detail}")
        
        # GPU Vendor Summary
        report.append(f"\n  GPU Vendor: {self.gpu_vendor}")
        if self.gpu_vendor == 'None':
            report.append("  Note: No discrete GPU detected. System will use CPU for AI workloads.")
        
        report.append("\n" + "=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        report_text = '\n'.join(report)
        
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        print(f"  [OK] Text report saved: {output_path}")
        return report_text
    
    def save_json_report(self, output_path='gpu_report.json'):
        """Save results as JSON"""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"  [OK] JSON report saved: {output_path}")
        
    def run_all_tests(self):
        """Run complete desktop testing suite for AI workloads"""
        print("\n" + "=" * 60)
        print("AI DESKTOP TESTING & ANALYSIS SUITE")
        print("   College AI Lab - Comprehensive Assessment")
        print("=" * 60 + "\n")
        
        # Print system detection info
        print(f"[INFO] Operating System: {platform.system()} {platform.release()}")
        print(f"[INFO] Architecture: {platform.machine()}")
        print(f"[INFO] Python: {platform.python_version()}")
        print(f"[INFO] GPU Vendor Detected: {self.gpu_vendor}")
        
        # Check if running remotely
        if IS_LINUX or IS_MACOS:
            if 'SSH_CLIENT' in os.environ or 'SSH_TTY' in os.environ:
                print("[INFO] Running via SSH/Remote session")
        
        # For NVIDIA, verify driver is working
        if self.gpu_vendor == 'NVIDIA':
            try:
                if IS_WINDOWS:
                    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10,
                                           creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0)
                else:
                    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
                if result.returncode != 0:
                    print("[WARNING] nvidia-smi failed - NVIDIA driver may have issues")
                else:
                    print("[OK] NVIDIA driver responding")
            except Exception as e:
                print(f"[WARNING] Could not verify NVIDIA driver: {e}")
        elif self.gpu_vendor == 'AMD':
            print("[OK] AMD GPU detected")
            if TORCH_ROCM_AVAILABLE:
                print("[OK] PyTorch ROCm support available")
        elif self.gpu_vendor == 'Apple':
            print("[OK] Apple Silicon detected")
            if TORCH_MPS_AVAILABLE:
                print("[OK] PyTorch MPS support available")
        elif self.gpu_vendor == 'Intel':
            print("[OK] Intel GPU detected")
        else:
            print("[INFO] No discrete GPU detected - running CPU-only tests")
        
        print("")
        
        # Collect information
        self.collect_system_info()
        self.collect_gpu_info()
        self.collect_memory_info()
        self.collect_storage_info()
        self.collect_network_info()
        self.collect_software_stack()
        
        # Run tests
        self.run_performance_tests()
        self.run_stress_tests()
        self.run_health_checks()
        
        # Analyze
        self.analyze_suitability()
        
        # Generate reports
        print("\n[INFO] Generating Reports...")
        
        # Create output directory
        output_dir = Path('ai_desktop_test_results')
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Generate all reports
        self.generate_report_image(output_dir / f'ai_report_{timestamp}.png')
        report_text = self.generate_text_report(output_dir / f'ai_report_{timestamp}.txt')
        self.save_json_report(output_dir / f'ai_report_{timestamp}.json')
        
        # Also save to current directory with simple names
        self.generate_report_image('ai_desktop_report.png')
        self.save_json_report('ai_desktop_report.json')
        
        # Save text report to current directory too
        with open('ai_desktop_report.txt', 'w') as f:
            f.write(report_text)
        
        # Print summary to console
        print("\n" + report_text)
        
        print("\n" + "=" * 60)
        print("[COMPLETE] AI DESKTOP TESTING COMPLETE")
        print("=" * 60)
        print(f"\nReports saved to:")
        print(f"   - ai_desktop_report.png (Visual Report)")
        print(f"   - ai_desktop_report.txt (Text Report)")
        print(f"   - ai_desktop_report.json (Machine-readable)")
        print(f"   - ai_desktop_test_results/ (Timestamped archives)")
        
        return True


def main():
    """Main entry point"""
    tester = GPUTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
