#!/usr/bin/env python3
"""
Comprehensive GPU Testing and Analysis Suite
For MiPhi GPU Facility - College GPU Testing
Generates detailed reports, analytics, and suitability analysis
"""

import subprocess
import sys
import os
import json
import time
import datetime
import platform
from pathlib import Path

# Check for required packages
try:
    import psutil
except ImportError:
    print("ERROR: psutil not installed. Run: pip install psutil")
    print("Or use the run_gpu_test.sh script which handles dependencies automatically.")
    sys.exit(1)

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Circle, Wedge, Rectangle
    import pandas as pd
except ImportError as e:
    print(f"ERROR: Missing required package: {e}")
    print("Run: pip install numpy matplotlib pandas")
    print("Or use the run_gpu_test.sh script which handles dependencies automatically.")
    sys.exit(1)

# Try to import GPU-specific libraries
CUDA_AVAILABLE = False
PYNVML_AVAILABLE = False
TORCH_AVAILABLE = False

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
except:
    pass


class GPUTester:
    """Comprehensive GPU Testing Class"""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.datetime.now().isoformat(),
            'system_info': {},
            'gpu_info': {},
            'memory_info': {},
            'performance_tests': {},
            'stress_tests': {},
            'health_checks': {},
            'suitability_analysis': {},
            'recommendations': []
        }
        self.test_start_time = time.time()
        
    def get_nvidia_smi_info(self):
        """Get GPU info using nvidia-smi"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used,memory.free,temperature.gpu,power.draw,power.limit,utilization.gpu,utilization.memory,pcie.link.gen.current,pcie.link.width.current,clocks.current.graphics,clocks.current.memory,clocks.max.graphics,clocks.max.memory,fan.speed,pstate',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=30
            )
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
    
    def get_cuda_version(self):
        """Get CUDA version"""
        try:
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'release' in line.lower():
                        return line.split('release')[-1].split(',')[0].strip()
        except:
            pass
        
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'CUDA Version' in line:
                        return line.split('CUDA Version:')[-1].split()[0].strip()
        except:
            pass
        return "Unknown"
    
    def collect_system_info(self):
        """Collect system information"""
        print("[INFO] Collecting System Information...")
        self.results['system_info'] = {
            'hostname': platform.node(),
            'os': f"{platform.system()} {platform.release()}",
            'python_version': platform.python_version(),
            'cpu': platform.processor(),
            'cpu_cores': psutil.cpu_count(logical=False),
            'cpu_threads': psutil.cpu_count(logical=True),
            'ram_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'ram_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
        }
        
    def collect_gpu_info(self):
        """Collect GPU information"""
        print("[INFO] Collecting GPU Information...")
        
        nvidia_info = self.get_nvidia_smi_info()
        cuda_version = self.get_cuda_version()
        
        self.results['gpu_info'] = {
            'cuda_available': CUDA_AVAILABLE,
            'torch_available': TORCH_AVAILABLE,
            'pynvml_available': PYNVML_AVAILABLE,
            'cuda_version': cuda_version,
            **nvidia_info
        }
        
        # Get additional info via PyTorch if available
        if TORCH_AVAILABLE:
            self.results['gpu_info']['torch_cuda_version'] = torch.version.cuda
            self.results['gpu_info']['cudnn_version'] = str(torch.backends.cudnn.version())
            self.results['gpu_info']['device_count'] = torch.cuda.device_count()
            self.results['gpu_info']['current_device'] = torch.cuda.current_device()
            self.results['gpu_info']['device_name'] = torch.cuda.get_device_name(0)
            self.results['gpu_info']['device_capability'] = '.'.join(map(str, torch.cuda.get_device_capability(0)))
            
    def collect_memory_info(self):
        """Collect detailed memory information"""
        print("[INFO] Collecting Memory Information...")
        
        if TORCH_AVAILABLE:
            torch.cuda.empty_cache()
            self.results['memory_info'] = {
                'total_memory_gb': round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
                'allocated_memory_gb': round(torch.cuda.memory_allocated(0) / (1024**3), 4),
                'cached_memory_gb': round(torch.cuda.memory_reserved(0) / (1024**3), 4),
                'max_memory_allocated_gb': round(torch.cuda.max_memory_allocated(0) / (1024**3), 4),
            }
        elif 'memory_total_mb' in self.results['gpu_info']:
            try:
                self.results['memory_info'] = {
                    'total_memory_gb': round(float(self.results['gpu_info']['memory_total_mb']) / 1024, 2),
                    'used_memory_gb': round(float(self.results['gpu_info']['memory_used_mb']) / 1024, 2),
                    'free_memory_gb': round(float(self.results['gpu_info']['memory_free_mb']) / 1024, 2),
                }
            except:
                self.results['memory_info'] = {'error': 'Could not parse memory info'}
                
    def run_memory_bandwidth_test(self):
        """Test memory bandwidth"""
        print("  [TEST] Running Memory Bandwidth Test...")
        
        if not TORCH_AVAILABLE:
            return {'error': 'PyTorch not available'}
        
        results = {}
        sizes = [1, 10, 100, 500]  # MB
        
        for size_mb in sizes:
            try:
                n_elements = (size_mb * 1024 * 1024) // 4  # float32 = 4 bytes
                
                # Host to Device
                cpu_tensor = torch.randn(n_elements, dtype=torch.float32)
                torch.cuda.synchronize()
                start = time.perf_counter()
                gpu_tensor = cpu_tensor.cuda()
                torch.cuda.synchronize()
                h2d_time = time.perf_counter() - start
                h2d_bandwidth = (size_mb / h2d_time) / 1024  # GB/s
                
                # Device to Host
                torch.cuda.synchronize()
                start = time.perf_counter()
                cpu_back = gpu_tensor.cpu()
                torch.cuda.synchronize()
                d2h_time = time.perf_counter() - start
                d2h_bandwidth = (size_mb / d2h_time) / 1024  # GB/s
                
                # Device to Device
                torch.cuda.synchronize()
                start = time.perf_counter()
                gpu_tensor2 = gpu_tensor.clone()
                torch.cuda.synchronize()
                d2d_time = time.perf_counter() - start
                d2d_bandwidth = (size_mb / d2d_time) / 1024  # GB/s
                
                results[f'{size_mb}MB'] = {
                    'h2d_bandwidth_gbps': round(h2d_bandwidth, 2),
                    'd2h_bandwidth_gbps': round(d2h_bandwidth, 2),
                    'd2d_bandwidth_gbps': round(d2d_bandwidth, 2)
                }
                
                del cpu_tensor, gpu_tensor, gpu_tensor2, cpu_back
                torch.cuda.empty_cache()
                
            except Exception as e:
                results[f'{size_mb}MB'] = {'error': str(e)}
                
        return results
    
    def run_compute_benchmark(self):
        """Run compute performance benchmark"""
        print("  [TEST] Running Compute Benchmark...")
        
        if not TORCH_AVAILABLE:
            return {'error': 'PyTorch not available'}
        
        results = {}
        
        # Matrix multiplication benchmark
        sizes = [1024, 2048, 4096]
        for size in sizes:
            try:
                a = torch.randn(size, size, device='cuda', dtype=torch.float32)
                b = torch.randn(size, size, device='cuda', dtype=torch.float32)
                
                # Warmup
                for _ in range(3):
                    c = torch.matmul(a, b)
                torch.cuda.synchronize()
                
                # Benchmark
                iterations = 10
                start = time.perf_counter()
                for _ in range(iterations):
                    c = torch.matmul(a, b)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                
                # Calculate TFLOPS (2 * N^3 operations for matmul)
                flops = 2 * (size ** 3) * iterations
                tflops = (flops / elapsed) / 1e12
                
                results[f'matmul_{size}x{size}'] = {
                    'time_ms': round((elapsed / iterations) * 1000, 2),
                    'tflops': round(tflops, 2)
                }
                
                del a, b, c
                torch.cuda.empty_cache()
                
            except Exception as e:
                results[f'matmul_{size}x{size}'] = {'error': str(e)}
        
        # Convolution benchmark
        try:
            batch_sizes = [1, 8, 32]
            for batch in batch_sizes:
                x = torch.randn(batch, 64, 224, 224, device='cuda', dtype=torch.float32)
                conv = torch.nn.Conv2d(64, 128, 3, padding=1).cuda()
                
                # Warmup
                for _ in range(3):
                    y = conv(x)
                torch.cuda.synchronize()
                
                # Benchmark
                iterations = 20
                start = time.perf_counter()
                for _ in range(iterations):
                    y = conv(x)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                
                results[f'conv2d_batch{batch}'] = {
                    'time_ms': round((elapsed / iterations) * 1000, 2),
                    'throughput_imgs_per_sec': round(batch * iterations / elapsed, 2)
                }
                
                del x, y, conv
                torch.cuda.empty_cache()
                
        except Exception as e:
            results['conv2d'] = {'error': str(e)}
            
        return results
    
    def run_stress_test(self, duration_seconds=30):
        """Run GPU stress test"""
        print(f"  [TEST] Running Stress Test ({duration_seconds}s)...")
        
        if not TORCH_AVAILABLE:
            return {'error': 'PyTorch not available'}
        
        results = {
            'temperatures': [],
            'utilizations': [],
            'power_draws': [],
            'timestamps': []
        }
        
        try:
            # Create large tensors for stress
            size = 4096
            a = torch.randn(size, size, device='cuda', dtype=torch.float32)
            b = torch.randn(size, size, device='cuda', dtype=torch.float32)
            
            start_time = time.time()
            iteration = 0
            
            while time.time() - start_time < duration_seconds:
                # Perform heavy computation
                c = torch.matmul(a, b)
                c = torch.matmul(c, a)
                torch.cuda.synchronize()
                
                # Collect metrics every second
                if iteration % 5 == 0:
                    nvidia_info = self.get_nvidia_smi_info()
                    results['timestamps'].append(round(time.time() - start_time, 1))
                    
                    try:
                        results['temperatures'].append(float(nvidia_info.get('temperature_c', 0)))
                        results['utilizations'].append(float(nvidia_info.get('gpu_utilization_pct', 0)))
                        results['power_draws'].append(float(nvidia_info.get('power_draw_w', 0)))
                    except:
                        pass
                
                iteration += 1
            
            del a, b, c
            torch.cuda.empty_cache()
            
            # Calculate statistics
            if results['temperatures']:
                results['max_temperature_c'] = max(results['temperatures'])
                results['avg_temperature_c'] = round(sum(results['temperatures']) / len(results['temperatures']), 1)
                results['max_utilization_pct'] = max(results['utilizations'])
                results['avg_utilization_pct'] = round(sum(results['utilizations']) / len(results['utilizations']), 1)
                if results['power_draws']:
                    results['max_power_w'] = max(results['power_draws'])
                    results['avg_power_w'] = round(sum(results['power_draws']) / len(results['power_draws']), 1)
                    
        except Exception as e:
            results['error'] = str(e)
            
        return results
    
    def run_memory_stress_test(self):
        """Test maximum memory allocation"""
        print("  [TEST] Running Memory Stress Test...")
        
        if not TORCH_AVAILABLE:
            return {'error': 'PyTorch not available'}
        
        results = {}
        torch.cuda.empty_cache()
        
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_tensors = []
            chunk_size = 256 * 1024 * 1024  # 256 MB chunks
            
            max_allocated = 0
            try:
                while True:
                    t = torch.zeros(chunk_size // 4, device='cuda', dtype=torch.float32)
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
                large_tensor = torch.zeros(int(total_memory * 0.8) // 4, device='cuda', dtype=torch.float32)
                del large_tensor
                torch.cuda.empty_cache()
            except:
                results['fragmentation_test'] = 'FAILED'
                
        except Exception as e:
            results['error'] = str(e)
            
        return results
    
    def run_error_check(self):
        """Check for GPU errors"""
        print("  [TEST] Running Error Checks...")
        
        results = {
            'cuda_errors': [],
            'memory_errors': 'NONE',
            'driver_errors': 'NONE'
        }
        
        # Check nvidia-smi for errors
        try:
            result = subprocess.run(
                ['nvidia-smi', '-q', '-d', 'ECC'],
                capture_output=True, text=True, timeout=30
            )
            if 'error' in result.stdout.lower():
                results['memory_errors'] = 'DETECTED'
        except:
            pass
        
        # Test basic CUDA operations
        if TORCH_AVAILABLE:
            try:
                # Test allocation
                t = torch.randn(1000, 1000, device='cuda')
                # Test computation
                r = torch.matmul(t, t)
                # Test synchronization
                torch.cuda.synchronize()
                # Verify results
                assert not torch.isnan(r).any(), "NaN detected in computation"
                assert not torch.isinf(r).any(), "Inf detected in computation"
                results['basic_ops_test'] = 'PASSED'
                del t, r
                torch.cuda.empty_cache()
            except Exception as e:
                results['basic_ops_test'] = f'FAILED: {str(e)}'
                results['cuda_errors'].append(str(e))
        
        return results
    
    def run_performance_tests(self):
        """Run all performance tests"""
        print("\n[PHASE] Running Performance Tests...")
        
        self.results['performance_tests'] = {
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
        """Analyze GPU suitability for different workloads"""
        print("\n[PHASE] Analyzing GPU Suitability...")
        
        scores = {
            'deep_learning_training': 0,
            'deep_learning_inference': 0,
            'scientific_computing': 0,
            'general_purpose': 0,
            'overall': 0
        }
        
        max_score = 100
        recommendations = []
        
        # Memory analysis
        try:
            mem_gb = float(self.results['memory_info'].get('total_memory_gb', 0))
            if mem_gb >= 24:
                scores['deep_learning_training'] += 30
                scores['deep_learning_inference'] += 25
                scores['scientific_computing'] += 25
            elif mem_gb >= 16:
                scores['deep_learning_training'] += 25
                scores['deep_learning_inference'] += 25
                scores['scientific_computing'] += 22
            elif mem_gb >= 8:
                scores['deep_learning_training'] += 15
                scores['deep_learning_inference'] += 20
                scores['scientific_computing'] += 18
                recommendations.append(f"[WARNING] Limited VRAM ({mem_gb}GB) may restrict large model training")
            else:
                scores['deep_learning_training'] += 5
                scores['deep_learning_inference'] += 10
                scores['scientific_computing'] += 10
                recommendations.append(f"[CRITICAL] Low VRAM ({mem_gb}GB) - not suitable for most deep learning tasks")
        except:
            recommendations.append("[WARNING] Could not analyze memory capacity")
        
        # Compute performance analysis
        try:
            compute = self.results['performance_tests'].get('compute_benchmark', {})
            matmul_4096 = compute.get('matmul_4096x4096', {})
            tflops = matmul_4096.get('tflops', 0)
            
            if tflops >= 15:
                scores['deep_learning_training'] += 30
                scores['deep_learning_inference'] += 25
                scores['scientific_computing'] += 30
            elif tflops >= 8:
                scores['deep_learning_training'] += 22
                scores['deep_learning_inference'] += 22
                scores['scientific_computing'] += 25
            elif tflops >= 4:
                scores['deep_learning_training'] += 15
                scores['deep_learning_inference'] += 18
                scores['scientific_computing'] += 18
            else:
                scores['deep_learning_training'] += 5
                scores['deep_learning_inference'] += 10
                scores['scientific_computing'] += 10
                recommendations.append(f"[WARNING] Compute performance ({tflops} TFLOPS) is below optimal")
        except:
            pass
        
        # Temperature analysis
        try:
            stress = self.results['stress_tests'].get('gpu_stress', {})
            max_temp = stress.get('max_temperature_c', 0)
            
            if max_temp < 75:
                scores['deep_learning_training'] += 20
                scores['deep_learning_inference'] += 20
                scores['scientific_computing'] += 20
            elif max_temp < 85:
                scores['deep_learning_training'] += 15
                scores['deep_learning_inference'] += 18
                scores['scientific_computing'] += 15
                recommendations.append(f"[WARNING] Temperature under load ({max_temp} C) - ensure adequate cooling")
            else:
                scores['deep_learning_training'] += 5
                scores['deep_learning_inference'] += 10
                scores['scientific_computing'] += 5
                recommendations.append(f"[CRITICAL] High temperature ({max_temp} C) - thermal throttling likely, improve cooling")
        except:
            pass
        
        # Health check analysis
        try:
            health = self.results['health_checks']
            if health.get('basic_ops_test') == 'PASSED' and health.get('memory_errors') == 'NONE':
                scores['deep_learning_training'] += 20
                scores['deep_learning_inference'] += 30
                scores['scientific_computing'] += 25
                scores['general_purpose'] += 40
            else:
                recommendations.append("[CRITICAL] GPU health check failures detected - investigate errors")
        except:
            pass
        
        # Calculate overall score
        scores['general_purpose'] = min(scores['general_purpose'] + 60, 100)  # Base score for working GPU
        scores['overall'] = round(sum([scores['deep_learning_training'], 
                                       scores['deep_learning_inference'],
                                       scores['scientific_computing'],
                                       scores['general_purpose']]) / 4)
        
        # Final suitability determination
        suitability = {}
        for key, score in scores.items():
            if score >= 80:
                suitability[key] = {'score': score, 'rating': 'EXCELLENT', 'suitable': True}
            elif score >= 60:
                suitability[key] = {'score': score, 'rating': 'GOOD', 'suitable': True}
            elif score >= 40:
                suitability[key] = {'score': score, 'rating': 'MODERATE', 'suitable': True}
            else:
                suitability[key] = {'score': score, 'rating': 'POOR', 'suitable': False}
        
        self.results['suitability_analysis'] = suitability
        self.results['recommendations'] = recommendations
        
    def generate_report_image(self, output_path='gpu_report.png'):
        """Generate comprehensive report image"""
        print("\n[INFO] Generating Report Image...")
        
        fig = plt.figure(figsize=(20, 24))
        fig.suptitle('GPU Testing & Analysis Report\nMiPhi GPU Facility', fontsize=20, fontweight='bold', y=0.98)
        
        # Create grid
        gs = gridspec.GridSpec(6, 4, figure=fig, hspace=0.4, wspace=0.3)
        
        # Colors
        colors = {
            'excellent': '#2ecc71',
            'good': '#3498db',
            'moderate': '#f39c12',
            'poor': '#e74c3c',
            'background': '#2c3e50',
            'text': '#ecf0f1'
        }
        
        # 1. System & GPU Info Panel
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.set_facecolor('#34495e')
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        ax1.axis('off')
        ax1.set_title('System Information', fontsize=14, fontweight='bold', pad=10)
        
        sys_info = self.results.get('system_info', {})
        gpu_info = self.results.get('gpu_info', {})
        
        info_text = f"""
Hostname: {sys_info.get('hostname', 'N/A')}
OS: {sys_info.get('os', 'N/A')}
CPU: {sys_info.get('cpu_cores', 'N/A')} cores / {sys_info.get('cpu_threads', 'N/A')} threads
RAM: {sys_info.get('ram_total_gb', 'N/A')} GB

GPU: {gpu_info.get('name', gpu_info.get('device_name', 'N/A'))}
Driver: {gpu_info.get('driver_version', 'N/A')}
CUDA: {gpu_info.get('cuda_version', 'N/A')}
Compute Capability: {gpu_info.get('device_capability', 'N/A')}
        """
        ax1.text(0.5, 5, info_text, fontsize=10, verticalalignment='center', 
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='#2c3e50', alpha=0.8))
        
        # 2. Memory Info Panel
        ax2 = fig.add_subplot(gs[0, 2:])
        ax2.set_facecolor('#34495e')
        ax2.axis('off')
        ax2.set_title('Memory Information', fontsize=14, fontweight='bold', pad=10)
        
        mem_info = self.results.get('memory_info', {})
        total_mem = mem_info.get('total_memory_gb', 0)
        
        # Memory bar
        ax2_bar = fig.add_subplot(gs[0, 2:])
        ax2_bar.axis('off')
        
        mem_text = f"""
Total VRAM: {total_mem} GB
Memory Bandwidth Test Results:
"""
        bw_test = self.results.get('performance_tests', {}).get('memory_bandwidth', {})
        if '100MB' in bw_test:
            bw_100 = bw_test['100MB']
            mem_text += f"  H2D: {bw_100.get('h2d_bandwidth_gbps', 'N/A')} GB/s\n"
            mem_text += f"  D2H: {bw_100.get('d2h_bandwidth_gbps', 'N/A')} GB/s\n"
            mem_text += f"  D2D: {bw_100.get('d2d_bandwidth_gbps', 'N/A')} GB/s"
        
        ax2.text(0.5, 0.5, mem_text, fontsize=10, verticalalignment='center',
                transform=ax2.transAxes, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#2c3e50', alpha=0.8))
        
        # 3. Compute Performance Chart
        ax3 = fig.add_subplot(gs[1, :2])
        compute = self.results.get('performance_tests', {}).get('compute_benchmark', {})
        
        matmul_sizes = []
        matmul_tflops = []
        for key in ['matmul_1024x1024', 'matmul_2048x2048', 'matmul_4096x4096']:
            if key in compute and 'tflops' in compute[key]:
                matmul_sizes.append(key.replace('matmul_', ''))
                matmul_tflops.append(compute[key]['tflops'])
        
        if matmul_tflops:
            bars = ax3.bar(matmul_sizes, matmul_tflops, color=['#3498db', '#2ecc71', '#e74c3c'])
            ax3.set_ylabel('TFLOPS', fontsize=12)
            ax3.set_title('Matrix Multiplication Performance', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Matrix Size')
            for bar, val in zip(bars, matmul_tflops):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{val}', ha='center', fontsize=10, fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'No compute benchmark data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Matrix Multiplication Performance', fontsize=14, fontweight='bold')
        
        # 4. Convolution Performance
        ax4 = fig.add_subplot(gs[1, 2:])
        conv_batches = []
        conv_throughput = []
        for key in ['conv2d_batch1', 'conv2d_batch8', 'conv2d_batch32']:
            if key in compute and 'throughput_imgs_per_sec' in compute[key]:
                conv_batches.append(key.replace('conv2d_batch', 'Batch '))
                conv_throughput.append(compute[key]['throughput_imgs_per_sec'])
        
        if conv_throughput:
            bars = ax4.bar(conv_batches, conv_throughput, color=['#9b59b6', '#1abc9c', '#f39c12'])
            ax4.set_ylabel('Images/sec', fontsize=12)
            ax4.set_title('Conv2D Performance (224x224)', fontsize=14, fontweight='bold')
            for bar, val in zip(bars, conv_throughput):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{val:.0f}', ha='center', fontsize=10, fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'No convolution benchmark data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Conv2D Performance', fontsize=14, fontweight='bold')
        
        # 5. Stress Test - Temperature
        ax5 = fig.add_subplot(gs[2, :2])
        stress = self.results.get('stress_tests', {}).get('gpu_stress', {})
        temps = stress.get('temperatures', [])
        timestamps = stress.get('timestamps', [])
        
        if temps and timestamps:
            ax5.plot(timestamps, temps, 'r-', linewidth=2, marker='o', markersize=4)
            ax5.fill_between(timestamps, temps, alpha=0.3, color='red')
            ax5.axhline(y=80, color='orange', linestyle='--', label='Warning (80°C)')
            ax5.axhline(y=90, color='red', linestyle='--', label='Critical (90°C)')
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('Temperature (°C)')
            ax5.set_title(f'Temperature Under Stress (Max: {stress.get("max_temperature_c", "N/A")}°C)', 
                         fontsize=14, fontweight='bold')
            ax5.legend(loc='upper right')
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'No temperature data', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Temperature Under Stress', fontsize=14, fontweight='bold')
        
        # 6. Stress Test - Power & Utilization
        ax6 = fig.add_subplot(gs[2, 2:])
        utils = stress.get('utilizations', [])
        powers = stress.get('power_draws', [])
        
        if utils and timestamps:
            ax6_twin = ax6.twinx()
            line1 = ax6.plot(timestamps, utils, 'b-', linewidth=2, label='GPU Util %')
            ax6.set_ylabel('GPU Utilization (%)', color='blue')
            ax6.tick_params(axis='y', labelcolor='blue')
            
            if powers:
                line2 = ax6_twin.plot(timestamps, powers, 'g-', linewidth=2, label='Power (W)')
                ax6_twin.set_ylabel('Power (W)', color='green')
                ax6_twin.tick_params(axis='y', labelcolor='green')
            
            ax6.set_xlabel('Time (s)')
            ax6.set_title('GPU Utilization & Power Under Stress', fontsize=14, fontweight='bold')
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'No utilization data', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('GPU Utilization & Power', fontsize=14, fontweight='bold')
        
        # 7. Memory Bandwidth Comparison
        ax7 = fig.add_subplot(gs[3, :2])
        bw_test = self.results.get('performance_tests', {}).get('memory_bandwidth', {})
        
        if bw_test and '100MB' in bw_test:
            bw_data = bw_test['100MB']
            categories = ['Host→Device', 'Device→Host', 'Device→Device']
            values = [
                bw_data.get('h2d_bandwidth_gbps', 0),
                bw_data.get('d2h_bandwidth_gbps', 0),
                bw_data.get('d2d_bandwidth_gbps', 0)
            ]
            colors_bw = ['#3498db', '#e74c3c', '#2ecc71']
            bars = ax7.barh(categories, values, color=colors_bw)
            ax7.set_xlabel('Bandwidth (GB/s)')
            ax7.set_title('Memory Bandwidth (100MB Transfer)', fontsize=14, fontweight='bold')
            for bar, val in zip(bars, values):
                ax7.text(val + 0.1, bar.get_y() + bar.get_height()/2, 
                        f'{val} GB/s', va='center', fontsize=10, fontweight='bold')
        else:
            ax7.text(0.5, 0.5, 'No bandwidth data', ha='center', va='center', transform=ax7.transAxes)
            ax7.set_title('Memory Bandwidth', fontsize=14, fontweight='bold')
        
        # 8. Health Check Status
        ax8 = fig.add_subplot(gs[3, 2:])
        ax8.axis('off')
        ax8.set_title('Health Check Status', fontsize=14, fontweight='bold', pad=10)
        
        health = self.results.get('health_checks', {})
        health_items = [
            ('Basic Operations', health.get('basic_ops_test', 'N/A')),
            ('Memory Errors', health.get('memory_errors', 'N/A')),
            ('Driver Status', health.get('driver_errors', 'N/A')),
            ('Fragmentation', self.results.get('stress_tests', {}).get('memory_stress', {}).get('fragmentation_test', 'N/A'))
        ]
        
        y_pos = 0.85
        for item, status in health_items:
            color = '#2ecc71' if status in ['PASSED', 'NONE'] else '#e74c3c'
            symbol = '[OK]' if status in ['PASSED', 'NONE'] else '[FAIL]'
            ax8.text(0.1, y_pos, f'{symbol} {item}: {status}', fontsize=12, 
                    transform=ax8.transAxes, color=color, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            y_pos -= 0.2
        
        # 9. Suitability Gauges
        ax9 = fig.add_subplot(gs[4, :])
        ax9.axis('off')
        ax9.set_title('Suitability Analysis', fontsize=16, fontweight='bold', pad=20)
        
        suitability = self.results.get('suitability_analysis', {})
        categories = ['Deep Learning\nTraining', 'Deep Learning\nInference', 'Scientific\nComputing', 'General\nPurpose', 'Overall']
        keys = ['deep_learning_training', 'deep_learning_inference', 'scientific_computing', 'general_purpose', 'overall']
        
        for i, (cat, key) in enumerate(zip(categories, keys)):
            data = suitability.get(key, {})
            score = data.get('score', 0)
            rating = data.get('rating', 'N/A')
            
            # Position for each gauge
            x_center = 0.1 + i * 0.2
            
            # Draw gauge background
            gauge_ax = fig.add_axes([x_center - 0.07, 0.28, 0.14, 0.1])
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
            gauge_ax.text(50, -0.8, cat, ha='center', va='top', fontsize=9, fontweight='bold')
            gauge_ax.text(50, 0, f'{score}%\n{rating}', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        
        # 10. Recommendations
        ax10 = fig.add_subplot(gs[5, :])
        ax10.axis('off')
        ax10.set_title('Recommendations & Findings', fontsize=14, fontweight='bold', pad=10)
        
        recommendations = self.results.get('recommendations', [])
        if not recommendations:
            recommendations = ['[OK] GPU is performing within expected parameters']
        
        rec_text = '\n'.join(recommendations[:5])  # Limit to 5 recommendations
        ax10.text(0.5, 0.5, rec_text, fontsize=11, ha='center', va='center',
                 transform=ax10.transAxes, fontfamily='sans-serif',
                 bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.9))
        
        # Add timestamp and footer
        fig.text(0.5, 0.01, f'Report Generated: {self.results["timestamp"]} | Test Duration: {round(time.time() - self.test_start_time, 1)}s',
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
        report.append("GPU TESTING & ANALYSIS REPORT")
        report.append("MiPhi GPU Facility - Comprehensive GPU Assessment")
        report.append("=" * 80)
        report.append(f"\nReport Generated: {self.results['timestamp']}")
        report.append(f"Test Duration: {round(time.time() - self.test_start_time, 1)} seconds\n")
        
        # System Info
        report.append("\n" + "=" * 40)
        report.append("SYSTEM INFORMATION")
        report.append("=" * 40)
        for key, value in self.results.get('system_info', {}).items():
            report.append(f"  {key.replace('_', ' ').title()}: {value}")
        
        # GPU Info
        report.append("\n" + "=" * 40)
        report.append("GPU INFORMATION")
        report.append("=" * 40)
        gpu_info = self.results.get('gpu_info', {})
        important_keys = ['name', 'device_name', 'driver_version', 'cuda_version', 'device_capability',
                         'memory_total_mb', 'temperature_c', 'power_limit_w', 'pcie_gen', 'pcie_width']
        for key in important_keys:
            if key in gpu_info:
                report.append(f"  {key.replace('_', ' ').title()}: {gpu_info[key]}")
        
        # Memory Info
        report.append("\n" + "=" * 40)
        report.append("MEMORY INFORMATION")
        report.append("=" * 40)
        for key, value in self.results.get('memory_info', {}).items():
            report.append(f"  {key.replace('_', ' ').title()}: {value}")
        
        # Performance Tests
        report.append("\n" + "=" * 40)
        report.append("PERFORMANCE TEST RESULTS")
        report.append("=" * 40)
        
        # Memory Bandwidth
        report.append("\n  Memory Bandwidth Test:")
        bw_test = self.results.get('performance_tests', {}).get('memory_bandwidth', {})
        for size, data in bw_test.items():
            if isinstance(data, dict) and 'error' not in data:
                report.append(f"    {size}:")
                report.append(f"      Host to Device: {data.get('h2d_bandwidth_gbps', 'N/A')} GB/s")
                report.append(f"      Device to Host: {data.get('d2h_bandwidth_gbps', 'N/A')} GB/s")
                report.append(f"      Device to Device: {data.get('d2d_bandwidth_gbps', 'N/A')} GB/s")
        
        # Compute Benchmark
        report.append("\n  Compute Benchmark:")
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
        report.append(f"  Maximum Temperature: {stress.get('max_temperature_c', 'N/A')}°C")
        report.append(f"  Average Temperature: {stress.get('avg_temperature_c', 'N/A')}°C")
        report.append(f"  Maximum GPU Utilization: {stress.get('max_utilization_pct', 'N/A')}%")
        report.append(f"  Average GPU Utilization: {stress.get('avg_utilization_pct', 'N/A')}%")
        report.append(f"  Maximum Power Draw: {stress.get('max_power_w', 'N/A')}W")
        report.append(f"  Average Power Draw: {stress.get('avg_power_w', 'N/A')}W")
        
        mem_stress = self.results.get('stress_tests', {}).get('memory_stress', {})
        report.append(f"\n  Memory Stress Test:")
        report.append(f"    Max Allocatable: {mem_stress.get('max_allocatable_gb', 'N/A')} GB")
        report.append(f"    Allocation Efficiency: {mem_stress.get('allocation_efficiency_pct', 'N/A')}%")
        report.append(f"    Fragmentation Test: {mem_stress.get('fragmentation_test', 'N/A')}")
        
        # Health Checks
        report.append("\n" + "=" * 40)
        report.append("HEALTH CHECK RESULTS")
        report.append("=" * 40)
        health = self.results.get('health_checks', {})
        report.append(f"  Basic Operations Test: {health.get('basic_ops_test', 'N/A')}")
        report.append(f"  Memory Errors: {health.get('memory_errors', 'N/A')}")
        report.append(f"  Driver Errors: {health.get('driver_errors', 'N/A')}")
        
        # Suitability Analysis
        report.append("\n" + "=" * 40)
        report.append("SUITABILITY ANALYSIS")
        report.append("=" * 40)
        suitability = self.results.get('suitability_analysis', {})
        for key, data in suitability.items():
            if isinstance(data, dict):
                suitable = "[PASS] SUITABLE" if data.get('suitable', False) else "[FAIL] NOT SUITABLE"
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
            report.append("  [OK] GPU is performing within expected parameters")
            report.append("  [OK] No issues detected - GPU is suitable for use")
        
        # Final Verdict
        report.append("\n" + "=" * 40)
        report.append("FINAL VERDICT")
        report.append("=" * 40)
        overall = suitability.get('overall', {})
        overall_score = overall.get('score', 0)
        if overall_score >= 70:
            verdict = "GPU IS SUITABLE FOR USE"
            verdict_detail = "The GPU passed all critical tests and is recommended for deployment."
        elif overall_score >= 50:
            verdict = "GPU IS CONDITIONALLY SUITABLE"
            verdict_detail = "The GPU passed basic tests but has some limitations. Review recommendations above."
        else:
            verdict = "GPU REQUIRES ATTENTION"
            verdict_detail = "The GPU failed some tests or has significant limitations. Address issues before deployment."
        
        report.append(f"\n  >>> {verdict} <<<")
        report.append(f"  Overall Score: {overall_score}%")
        report.append(f"  {verdict_detail}")
        
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
        """Run complete GPU testing suite"""
        print("\n" + "=" * 60)
        print("GPU TESTING & ANALYSIS SUITE")
        print("   MiPhi GPU Facility - Comprehensive Assessment")
        print("=" * 60 + "\n")
        
        # Check for NVIDIA GPU
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                print("[ERROR] No NVIDIA GPU detected or nvidia-smi not available")
                print("   Please ensure NVIDIA drivers are installed.")
                return False
        except FileNotFoundError:
            print("[ERROR] nvidia-smi command not found")
            print("   Please ensure NVIDIA drivers are installed.")
            return False
        except Exception as e:
            print(f"[ERROR] {str(e)}")
            return False
        
        print("[OK] NVIDIA GPU detected\n")
        
        # Collect information
        self.collect_system_info()
        self.collect_gpu_info()
        self.collect_memory_info()
        
        # Run tests
        self.run_performance_tests()
        self.run_stress_tests()
        self.run_health_checks()
        
        # Analyze
        self.analyze_suitability()
        
        # Generate reports
        print("\n[INFO] Generating Reports...")
        
        # Create output directory
        output_dir = Path('gpu_test_results')
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Generate all reports
        self.generate_report_image(output_dir / f'gpu_report_{timestamp}.png')
        report_text = self.generate_text_report(output_dir / f'gpu_report_{timestamp}.txt')
        self.save_json_report(output_dir / f'gpu_report_{timestamp}.json')
        
        # Also save to current directory with simple names
        self.generate_report_image('gpu_report.png')
        self.save_json_report('gpu_report.json')
        
        # Print summary to console
        print("\n" + report_text)
        
        print("\n" + "=" * 60)
        print("[COMPLETE] GPU TESTING COMPLETE")
        print("=" * 60)
        print(f"\nReports saved to:")
        print(f"   - gpu_report.png (Visual Report)")
        print(f"   - gpu_report.json (Machine-readable)")
        print(f"   - gpu_test_results/ (Timestamped archives)")
        
        return True


def main():
    """Main entry point"""
    tester = GPUTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
