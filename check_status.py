#!/usr/bin/env python3
"""
Quick diagnostic script to check system status and identify timeout issues
"""

import os
import sys
import time
import subprocess
import requests
import logging
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking Dependencies...")
    print("-" * 40)

    deps = {
        'torch': 'PyTorch',
        'librosa': 'Librosa',
        'soundfile': 'SoundFile',
        'pydub': 'PyDub',
        'fastapi': 'FastAPI',
        'numpy': 'NumPy'
    }

    missing = []
    for module, name in deps.items():
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - MISSING")
            missing.append(name)

    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        return False
    else:
        print("\n✅ All dependencies available")
        return True

def check_ffmpeg():
    """Check if FFmpeg is available"""
    print("\n🎬 Checking FFmpeg...")
    print("-" * 40)

    try:
        result = subprocess.run(['ffmpeg', '-version'],
                               capture_output=True,
                               text=True,
                               timeout=5)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✅ FFmpeg available: {version_line}")
            return True
        else:
            print("❌ FFmpeg not working properly")
            return False
    except subprocess.TimeoutExpired:
        print("❌ FFmpeg check timed out")
        return False
    except FileNotFoundError:
        print("❌ FFmpeg not found in PATH")
        return False
    except Exception as e:
        print(f"❌ FFmpeg check failed: {e}")
        return False

def check_model_files():
    """Check if model files exist"""
    print("\n🤖 Checking Model Files...")
    print("-" * 40)

    base_dirs = [
        "data/models",
        "deployable_twi_speech_model",
        "model_package"
    ]

    found_models = []
    for base_dir in base_dirs:
        if os.path.exists(base_dir):
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    if file.endswith(('.pt', '.pth', '.pkl')):
                        found_models.append(os.path.join(root, file))

    if found_models:
        print(f"✅ Found {len(found_models)} model files:")
        for model in found_models[:5]:  # Show first 5
            size = os.path.getsize(model) / (1024*1024)  # MB
            print(f"   📄 {model} ({size:.1f} MB)")
        if len(found_models) > 5:
            print(f"   ... and {len(found_models) - 5} more")
        return True
    else:
        print("❌ No model files found")
        return False

def check_api_status():
    """Check if API is running"""
    print("\n🌐 Checking API Status...")
    print("-" * 40)

    ports = [8000, 8080, 5000]

    for port in ports:
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ API running on port {port}")
                try:
                    data = response.json()
                    print(f"   Status: {data.get('status', 'unknown')}")
                    print(f"   Uptime: {data.get('uptime', 'unknown')}")
                except:
                    print(f"   Response: {response.text[:100]}")
                return True
        except requests.exceptions.ConnectionError:
            print(f"❌ No API on port {port}")
        except requests.exceptions.Timeout:
            print(f"⏱️  API on port {port} timed out")
        except Exception as e:
            print(f"❌ Error checking port {port}: {e}")

    return False

def check_processes():
    """Check for running Python processes"""
    print("\n🔄 Checking Running Processes...")
    print("-" * 40)

    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True, timeout=10)
        lines = result.stdout.split('\n')

        python_processes = []
        for line in lines:
            if 'python' in line.lower() and ('app.py' in line or 'uvicorn' in line or 'api' in line):
                python_processes.append(line.strip())

        if python_processes:
            print(f"✅ Found {len(python_processes)} relevant Python processes:")
            for proc in python_processes[:3]:  # Show first 3
                parts = proc.split()
                if len(parts) >= 11:
                    pid = parts[1]
                    cpu = parts[2]
                    mem = parts[3]
                    cmd = ' '.join(parts[10:])[:60]
                    print(f"   PID {pid}: {cmd} (CPU: {cpu}%, MEM: {mem}%)")
        else:
            print("❌ No relevant Python processes found")

        return len(python_processes) > 0

    except Exception as e:
        print(f"❌ Error checking processes: {e}")
        return False

def check_disk_space():
    """Check available disk space"""
    print("\n💾 Checking Disk Space...")
    print("-" * 40)

    try:
        import shutil
        total, used, free = shutil.disk_usage('.')

        total_gb = total / (1024**3)
        used_gb = used / (1024**3)
        free_gb = free / (1024**3)
        used_percent = (used / total) * 100

        print(f"Total: {total_gb:.1f} GB")
        print(f"Used:  {used_gb:.1f} GB ({used_percent:.1f}%)")
        print(f"Free:  {free_gb:.1f} GB")

        if free_gb < 1.0:
            print("⚠️  Low disk space!")
            return False
        else:
            print("✅ Sufficient disk space")
            return True

    except Exception as e:
        print(f"❌ Error checking disk space: {e}")
        return False

def check_logs():
    """Check for recent log files"""
    print("\n📋 Checking Log Files...")
    print("-" * 40)

    log_dirs = ['logs', 'log', '.']
    log_files = []

    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            for file in os.listdir(log_dir):
                if file.endswith(('.log', '.out', '.err')):
                    log_path = os.path.join(log_dir, file)
                    if os.path.isfile(log_path):
                        mtime = os.path.getmtime(log_path)
                        log_files.append((log_path, mtime))

    if log_files:
        # Sort by modification time, newest first
        log_files.sort(key=lambda x: x[1], reverse=True)
        print(f"✅ Found {len(log_files)} log files:")

        for log_path, mtime in log_files[:3]:  # Show 3 most recent
            age = time.time() - mtime
            if age < 3600:  # Less than 1 hour
                age_str = f"{age/60:.0f}m ago"
            elif age < 86400:  # Less than 1 day
                age_str = f"{age/3600:.0f}h ago"
            else:
                age_str = f"{age/86400:.0f}d ago"

            size = os.path.getsize(log_path)
            print(f"   📄 {log_path} ({size} bytes, {age_str})")

        # Check most recent log for errors
        if log_files:
            recent_log = log_files[0][0]
            try:
                with open(recent_log, 'r') as f:
                    lines = f.readlines()
                    error_lines = [line for line in lines[-50:] if 'error' in line.lower() or 'timeout' in line.lower()]
                    if error_lines:
                        print(f"\n⚠️  Recent errors in {recent_log}:")
                        for line in error_lines[-3:]:  # Last 3 errors
                            print(f"     {line.strip()}")
            except Exception as e:
                print(f"   ❌ Could not read {recent_log}: {e}")

        return True
    else:
        print("❌ No log files found")
        return False

def run_diagnostic():
    """Run full diagnostic"""
    print("🔍 System Diagnostic Report")
    print("=" * 60)
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working Directory: {os.getcwd()}")
    print("=" * 60)

    checks = [
        ("Dependencies", check_dependencies),
        ("FFmpeg", check_ffmpeg),
        ("Model Files", check_model_files),
        ("API Status", check_api_status),
        ("Processes", check_processes),
        ("Disk Space", check_disk_space),
        ("Log Files", check_logs),
    ]

    results = {}

    for name, check_func in checks:
        try:
            result = check_func()
            results[name] = result
        except Exception as e:
            print(f"❌ Error in {name} check: {e}")
            results[name] = False

    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {name}")

    print(f"\nOverall: {passed}/{total} checks passed")

    if passed < total:
        print("\n⚠️  Issues detected. Check the details above.")
        print("\nCommon solutions:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Install FFmpeg: sudo apt install ffmpeg (Linux) or brew install ffmpeg (Mac)")
        print("3. Check if API is running: python app.py api")
        print("4. Check model files are in the correct location")
        return False
    else:
        print("\n✅ All checks passed! System appears healthy.")
        return True

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Usage: python check_status.py")
        print("       python check_status.py --quick    (skip some slower checks)")
        return

    success = run_diagnostic()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
