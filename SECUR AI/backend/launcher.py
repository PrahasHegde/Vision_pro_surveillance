#launcher.py
import subprocess
import time
import os
import signal
import sys

# --- CONFIGURATION: Absolute Path to your project ---
# CHANGE THIS to the exact path where your backend folder is!
# Example: "/home/pi/Desktop/SmartLock/backend"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# The list of scripts to run
scripts = [
    "server.py",       # Port 5001 (Data)
    "app.py",          # Port 5002 (Enrollment)
    "train_worker.py", # Training Worker
    "main_main.py"     # Port 5000 (Main System)
]

processes = []

def stop_all(signum, frame):
    print("\n[LAUNCHER] Stopping all services...")
    for p in processes:
        p.terminate()
    sys.exit(0)

# Handle Ctrl+C or System Stop signals
signal.signal(signal.SIGINT, stop_all)
signal.signal(signal.SIGTERM, stop_all)

print(f"[LAUNCHER] Starting Security System in {PROJECT_DIR}...")

# Start all scripts
for script in scripts:
    print(f" -> Launching {script}...")
    # using python3 explicitly and setting cwd ensures paths work
    p = subprocess.Popen(["python3", script], cwd=PROJECT_DIR)
    processes.append(p)
    time.sleep(2) # Wait 2s between starts to prevent port conflicts

print("[LAUNCHER] System Online. Press Ctrl+C to stop.")

# Keep the main script running to monitor children
try:
    while True:
        time.sleep(1)
        # Optional: Check if a child died and restart it?
        # For now, if the launcher dies, Systemd will restart the whole group.
except KeyboardInterrupt:
    stop_all(None, None)