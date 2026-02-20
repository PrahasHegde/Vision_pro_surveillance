# backend/server.py -> main code for running the backend server == main code for the project
#-------------------------------------------------------------------------------------------


# IMPORTS
import json
import os
import requests
import pickle
import shutil 
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from datetime import datetime
import logging


log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR) 

app = Flask(__name__)
# Enable CORS for all routes and origins
CORS(app, resources={r"/*": {"origins": "*"}})

# FORCE CORS HEADERS
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, DELETE"
    return response

# CONFIGURATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_DIR = os.path.join(BASE_DIR, "dataset") # Main dataset folder
PENDING_FILE = os.path.join(DATA_DIR, "pending_requests.json")
USERS_FILE = os.path.join(DATA_DIR, "user_details.json")
LOGS_FILE = os.path.join(DATA_DIR, "user_logs.json")
DB_FILE = os.path.join(BASE_DIR, "face_encodings.pickle") 
ENROLLMENT_SERVER_URL = "http://localhost:5002"
MAIN_SYSTEM_URL = "http://localhost:5000"

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True) 

for file_path in [PENDING_FILE, USERS_FILE, LOGS_FILE]:
    if not os.path.exists(file_path):
        with open(file_path, "w") as f: json.dump([], f)

def load_json(file_path):
    try:
        with open(file_path, "r") as f: return json.load(f)
    except: return []

def save_json(file_path, data):
    try:
        with open(file_path, "w") as f: json.dump(data, f, indent=4)
    except Exception as e:
        print(f"[ERROR] Failed to save {file_path}: {e}")


# ROUTES
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    identifier = data.get('identifier')
    password = data.get('password')
    
    users = load_json(USERS_FILE)
    
    if identifier == "admin" and password == "admin123":
         return jsonify({"status": "success", "user": {"name": "Administrator", "role": "admin"}})

    user = next((u for u in users if (u['username'] == identifier or u['email'] == identifier) and u['password'] == password), None)
    
    if user:
        return jsonify({"status": "success", "user": user})
    return jsonify({"status": "error", "message": "Invalid credentials"}), 401

@app.route('/check-availability', methods=['POST'])
def check_availability():
    try:
        data = request.json
        username = data.get("username", "").lower()
        email = data.get("email", "").lower()
        
        users = load_json(USERS_FILE)
        pending = load_json(PENDING_FILE)
        all_records = users + pending
        
        username_taken = any(u.get('username', '').lower() == username for u in all_records)
        email_taken = any(u.get('email', '').lower() == email for u in all_records)
                
        return jsonify({
            "username_available": not username_taken,
            "email_available": not email_taken
        })
    except Exception as e:
        print(f"Check Availability Error: {e}")
        return jsonify({"username_available": True, "email_available": True})

@app.route('/register-request', methods=['POST'])
def register_request():
    new_user = request.json
    pending = load_json(PENDING_FILE)
    users = load_json(USERS_FILE)
    
    if any(u['username'] == new_user['username'] for u in pending + users):
        return jsonify({"status": "error", "message": "Username taken"}), 400
        
    pending.append(new_user)
    save_json(PENDING_FILE, pending) 
    return jsonify({"status": "success", "message": "Request submitted"})

@app.route('/approve-user', methods=['POST'])
def approve_user():
    try:
        user_id = str(request.json.get("id"))
        pending = load_json(PENDING_FILE)
        user = next((u for u in pending if str(u["id"]) == user_id), None)
        
        if not user: return jsonify({"status": "error", "message": "User not found"}), 404

        # Trigger Training (Extracts images from temp video)
        try:
            response = requests.post(f"{ENROLLMENT_SERVER_URL}/process_pending_video", json={"username": user["username"]}, timeout=20)
            if response.status_code != 200: raise Exception(response.text)
        except Exception as e:
            print(f"Enrollment Error: {e}")
            return jsonify({"status": "error", "message": "Video processing failed"}), 500

        # Move to Active
        users = load_json(USERS_FILE)
        user["status"] = "active"
        user["registeredAt"] = datetime.now().isoformat()
        users.append(user)
        save_json(USERS_FILE, users)
        
        # Remove from Pending
        pending = [u for u in pending if str(u["id"]) != user_id]
        save_json(PENDING_FILE, pending)

        # Delete the Admin Video after approval
        try:
            video_path = os.path.join(DATASET_DIR, f"{user['username']}.mp4")
            if os.path.exists(video_path):
                os.remove(video_path)
                print(f"[CLEANUP] Deleted approved video: {video_path}")
        except Exception as e:
            print(f"[WARNING] Could not delete video: {e}")
        
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/deny-user', methods=['POST'])
def deny_user():
    try:
        user_id = str(request.json.get("id"))
        pending = load_json(PENDING_FILE)
        user = next((u for u in pending if str(u["id"]) == user_id), None)
        
        if user:
            try: requests.post(f"{ENROLLMENT_SERVER_URL}/delete_temp_video", json={"username": user["username"]})
            except: pass

            # Delete the Admin Video
            try:
                video_path = os.path.join(DATASET_DIR, f"{user['username']}.mp4")
                if os.path.exists(video_path):
                    os.remove(video_path)
                    print(f"[CLEANUP] Deleted denied video: {video_path}")
            except Exception as e:
                print(f"[ERROR] Could not delete video: {e}")

        pending = [u for u in pending if str(u["id"]) != user_id]
        save_json(PENDING_FILE, pending)
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/delete-user', methods=['POST'])
def delete_user():
    try:
        user_id = str(request.json.get('id'))
        users = load_json(USERS_FILE)
        
        user_to_delete = next((u for u in users if str(u['id']) == user_id), None)
        if not user_to_delete: return jsonify({"status": "error"}), 404
        
        username = user_to_delete['username']

        # CLEANUP DATASET (IMAGES)
        user_images_path = os.path.join(DATASET_DIR, username)
        if os.path.exists(user_images_path):
            try:
                shutil.rmtree(user_images_path) 
                print(f"[CLEANUP] Deleted image dataset for: {username}")
            except Exception as e:
                print(f"[ERROR] Failed to delete dataset folder: {e}")

        # CLEANUP VIDEO (Just in case it wasn't deleted before)
        user_video_path = os.path.join(DATASET_DIR, f"{username}.mp4")
        if os.path.exists(user_video_path):
            try:
                os.remove(user_video_path)
                print(f"[CLEANUP] Deleted registration video for: {username}")
            except: pass

        # REMOVE FROM JSON
        users = [u for u in users if str(u['id']) != user_id]
        save_json(USERS_FILE, users)
        
        # UPDATE PICKLE DB (SYNC)
        if os.path.exists(DB_FILE):
            try:
                with open(DB_FILE, "rb") as f: db = pickle.load(f)
                if username in db:
                    del db[username]
                    with open(DB_FILE, "wb") as f: pickle.dump(db, f)
                    try: requests.get(f"{MAIN_SYSTEM_URL}/reload_db", timeout=1)
                    except: pass
            except: pass

        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/get-all-data', methods=['GET'])
def get_all_data():
    return jsonify({
        "users": load_json(USERS_FILE),
        "pending": load_json(PENDING_FILE),
        "logs": load_json(LOGS_FILE)
    })

@app.route('/videos/<path:filename>')
def serve_video(filename):
    """Allows the Frontend to view user registration videos"""
    return send_from_directory(DATASET_DIR, filename)

if __name__ == '__main__':
    print(f"[INFO] Server Running on http://0.0.0.0:5001")
    app.run(host='0.0.0.0', port=5001, debug=True)
