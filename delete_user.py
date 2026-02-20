# delete_user.py -> code to remove user/embeddings from the pkl file == part of the main project code
#---------------------------------------------------------------------------------------------------


# IMPORTS
import pickle
import os
import sys


# CONFIGURATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "face_encodings.pickle")

def delete_user(username):
    # Check if DB exists
    if not os.path.exists(DB_FILE):
        print(f"[ERROR] Database file not found at: {DB_FILE}")
        return

    # Load the Database
    try:
        with open(DB_FILE, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"[ERROR] Could not read database: {e}")
        return

    # Check and Delete
    if username in data:
        del data[username]
        
        # Save changes
        with open(DB_FILE, "wb") as f:
            pickle.dump(data, f)
            
        print(f"[SUCCESS] User '{username}' has been deleted from the .pkl file.")
        print(f"Remaining users: {list(data.keys())}")
    else:
        print(f"[WARNING] User '{username}' was NOT found in the database.")
        print(f"Current users are: {list(data.keys())}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 delete_user.py <username>")
        print("Example: python3 delete_user.py test1")
    else:
        target_user = sys.argv[1]
        delete_user(target_user)
