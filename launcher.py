# launcher.py -> code used for running server, app, train_worker and main.py simultaneously == part of the main code for the project
#-----------------------------------------------------------------------------------------------------------------------------------



# IMPORTS
import subprocess
import time
import os
import signal
import sys


# PATH
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


# Scripts to run
scripts = {
    "server.py": None,       # Port 5001 (Data API)
    "app.py": None,          # Port 5002 (Video Processing)
    "train_worker.py": None, # Background Trainer
    "main.py": None     # Port 5000 (Main System)
}

def start_script(script_name):
    # print(f"[LAUNCHER] Starting {script_name}...")
    return subprocess.Popen([sys.executable, script_name], cwd=PROJECT_DIR)

def stop_all(signum, frame):
    # print("\n[LAUNCHER] Stopping all services...")
    for script, process in scripts.items():
        if process: process.terminate()
    sys.exit(0)

# Handle Stop Signals
signal.signal(signal.SIGINT, stop_all)
signal.signal(signal.SIGTERM, stop_all)

# print(f"[LAUNCHER] Initializing System in {PROJECT_DIR}...")

# Initial Start
for script in scripts:
    scripts[script] = start_script(script)
    time.sleep(2) # Stagger starts to prevent CPU spikes

# print("[LAUNCHER] System Online. Monitoring processes...")

# Monitor Loop (Restarts crashed scripts
try:
    while True:
        for script, process in scripts.items():
            if process.poll() is not None: # Process has died
                # print(f"[WARNING] {script} crashed! Restarting...")
                scripts[script] = start_script(script)
        time.sleep(5)
except KeyboardInterrupt:
    stop_all(None, None)
