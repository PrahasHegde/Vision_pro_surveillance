# backend/server.py
import json
import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)

# --- ABSOLUTE PATH FIX ---
# This ensures we always find the files relative to THIS script, not where you ran python from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

PENDING_FILE = os.path.join(DATA_DIR, "pending_requests.json")
USERS_FILE = os.path.join(DATA_DIR, "user_details.json")
LOGS_FILE = os.path.join(DATA_DIR, "user_logs.json")
ENROLLMENT_SERVER_URL = "http://localhost:5002" 

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
    with open(file_path, "w") as f: json.dump(data, f, indent=4)

# --- ROUTES ---
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    identifier = data.get('identifier')
    password = data.get('password')
    users = load_json(USERS_FILE)
    user = next((u for u in users if (u['username'] == identifier or u['email'] == identifier) and u['password'] == password), None)
    if user: return jsonify({"status": "success", "user": user})
    return jsonify({"status": "error", "message": "Invalid credentials"}), 401

@app.route('/register-request', methods=['POST'])
def register_request():
    new_user = request.json
    pending = load_json(PENDING_FILE)
    users = load_json(USERS_FILE)
    if any(u['username'] == new_user['username'] for u in pending) or \
       any(u['username'] == new_user['username'] for u in users):
        return jsonify({"status": "error", "message": "Username taken"}), 400
    pending.append(new_user)
    save_json(PENDING_FILE, pending)
    return jsonify({"status": "success", "message": "Request submitted"})

@app.route('/approve-user', methods=['POST'])
def approve_user():
    user_id = request.json.get("id")
    pending = load_json(PENDING_FILE)
    user = next((u for u in pending if u["id"] == user_id), None)
    if not user: return jsonify({"status": "error"}), 404

    # Trigger Processing
    try:
        response = requests.post(f"{ENROLLMENT_SERVER_URL}/process_pending_video", json={"username": user["username"]})
        if response.status_code != 200: raise Exception(response.text)
    except Exception as e:
        print(f"Error calling enrollment: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

    # Move to Active
    users = load_json(USERS_FILE)
    user["status"] = "active"
    user["registeredAt"] = datetime.now().isoformat()
    users.append(user)
    save_json(USERS_FILE, users)
    
    # Remove from Pending
    pending = [u for u in pending if u["id"] != user_id]
    save_json(PENDING_FILE, pending)
    
    return jsonify({"status": "success"})

@app.route('/deny-user', methods=['POST'])
def deny_user():
    user_id = request.json.get("id")
    pending = load_json(PENDING_FILE)
    user = next((u for u in pending if u["id"] == user_id), None)
    if not user: return jsonify({"status": "error"}), 404

    try: requests.post(f"{ENROLLMENT_SERVER_URL}/delete_temp_video", json={"username": user["username"]})
    except: pass

    pending = [u for u in pending if u["id"] != user_id]
    save_json(PENDING_FILE, pending)
    return jsonify({"status": "success"})

@app.route('/delete-user', methods=['POST'])
def delete_user():
    user_id = request.json.get('id')
    users = load_json(USERS_FILE)
    users = [u for u in users if u['id'] != user_id]
    save_json(USERS_FILE, users)
    return jsonify({"status": "success"})

@app.route('/get-all-data', methods=['GET'])
def get_all_data():
    return jsonify({"users": load_json(USERS_FILE), "pending": load_json(PENDING_FILE), "logs": load_json(LOGS_FILE)})

@app.route('/check-username', methods=['POST'])
def check_username():
    username = request.json.get("username")
    users = load_json(USERS_FILE)
    pending = load_json(PENDING_FILE)
    taken = any(u['username'] == username for u in users) or any(u['username'] == username for u in pending)
    return jsonify({"available": not taken})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)