# FILE: run_system.py
import subprocess
import time
import os
import sys

APP_SCRIPT = "app.py"
TRAIN_SCRIPT = "train_worker.py"
TRIGGER_FILE = "trigger_training.txt"

def main():
    # Clean start
    if os.path.exists(TRIGGER_FILE): os.remove(TRIGGER_FILE)

    while True:
        print("\n[BOSS] Starting Camera App (app.py)...")
        print("[BOSS] View at: http://<YOUR_PI_IP>:5000")
        
        # 1. Start the Camera App
        app_process = subprocess.Popen([sys.executable, APP_SCRIPT])
        
        # 2. Monitor loop
        while True:
            # Check if app crashed on its own
            if app_process.poll() is not None:
                print("[BOSS] App crashed! Restarting in 3s...")
                time.sleep(3)
                break 
            
            # Check for Training Trigger
            if os.path.exists(TRIGGER_FILE):
                print("\n[BOSS] Training Trigger Detected!")
                
                # A. Kill Camera App Safely
                print("[BOSS] Stopping Camera...")
                app_process.terminate()
                try:
                    app_process.wait(timeout=5)
                except:
                    app_process.kill()
                
                # B. Run Training Script
                print("[BOSS] Running Training Worker...")
                train_process = subprocess.run([sys.executable, TRAIN_SCRIPT])
                
                # C. Cleanup
                os.remove(TRIGGER_FILE)
                print("[BOSS] Training Done.")
                print("[BOSS] Restarting Camera in 2 seconds...")
                time.sleep(2)
                
                # Break inner loop to restart Camera App
                break
            
            time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[BOSS] Shutting down.")