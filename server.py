# backend/server.py
import json
import os
import requests
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
# Enable CORS for all routes and origins
CORS(app, resources={r"/*": {"origins": "*"}})

# --- 1. FORCE CORS HEADERS ---
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, DELETE"
    return response

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PENDING_FILE = os.path.join(DATA_DIR, "pending_requests.json")
USERS_FILE = os.path.join(DATA_DIR, "user_details.json")
LOGS_FILE = os.path.join(DATA_DIR, "user_logs.json")
DB_FILE = os.path.join(BASE_DIR, "face_encodings.pickle") 
ENROLLMENT_SERVER_URL = "http://localhost:5002"
MAIN_SYSTEM_URL = "http://localhost:5000"

# --- DEBUG: PRINT FILE LOCATIONS ---
print("="*50)
# print(f"[DEBUG] Backend Starting...")
# print(f"[DEBUG] Data Directory is located at: {DATA_DIR}")
# print(f"[DEBUG] Pending File: {PENDING_FILE}")
# print(f"[DEBUG] Users File:   {USERS_FILE}")
print("="*50)

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
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
        # Debug print to confirm save
        filename = os.path.basename(file_path)
        # print(f"[SUCCESS] Saved data to {filename} ({len(data)} records)")
    except Exception as e:
        print(f"[ERROR] Failed to save {file_path}: {e}")

# --- ROUTES ---

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    identifier = data.get('identifier')
    password = data.get('password')
    
    users = load_json(USERS_FILE)
    
    # Admin Fallback
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
    save_json(PENDING_FILE, pending) # This triggers the print statement
    return jsonify({"status": "success", "message": "Request submitted"})

@app.route('/approve-user', methods=['POST'])
def approve_user():
    try:
        user_id = str(request.json.get("id"))
        pending = load_json(PENDING_FILE)
        user = next((u for u in pending if str(u["id"]) == user_id), None)
        
        if not user: return jsonify({"status": "error", "message": "User not found"}), 404

        # 1. Trigger Training
        try:
            response = requests.post(f"{ENROLLMENT_SERVER_URL}/process_pending_video", json={"username": user["username"]}, timeout=20)
            if response.status_code != 200: raise Exception(response.text)
        except Exception as e:
            print(f"Enrollment Error: {e}")
            return jsonify({"status": "error", "message": "Video processing failed"}), 500

        # 2. Move to Active (Write to user_details.json)
        users = load_json(USERS_FILE)
        user["status"] = "active"
        user["registeredAt"] = datetime.now().isoformat()
        users.append(user)
        save_json(USERS_FILE, users)
        
        # 3. Remove from Pending (Write to pending_requests.json)
        pending = [u for u in pending if str(u["id"]) != user_id]
        save_json(PENDING_FILE, pending)
        
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
        
        users = [u for u in users if str(u['id']) != user_id]
        save_json(USERS_FILE, users)
        
        if os.path.exists(DB_FILE):
            try:
                with open(DB_FILE, "rb") as f: db = pickle.load(f)
                if user_to_delete['username'] in db:
                    del db[user_to_delete['username']]
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

if __name__ == '__main__':
    print(f"[INFO] Server Running on http://0.0.0.0:5001")
    app.run(host='0.0.0.0', port=5001, debug=True)