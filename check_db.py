# check_db.py -> code to check the number of users/embeddings in the .pkl file == part of the main code
#-----------------------------------------------------------------------------------------------


# IMPORTS
import pickle
import os

# PATH TO PKL FILE
DB_FILE = "face_encodings.pickle"

# MAIN
if os.path.exists(DB_FILE):
    print(f"--- CONTENT OF {DB_FILE} ---")
    try:
        with open(DB_FILE, "rb") as f:
            database = pickle.load(f)
            
        if not database:
            print("Database is empty.")
        else:
            for name, embedding in database.items():
                # Check if it's a list or single embedding
                dtype = type(embedding)
                print(f"User: {name:<20} | Type: {dtype}")
                
        print("-" * 30)
        print(f"Total Users: {len(database)}")
    except Exception as e:
        print(f"Error reading file: {e}")
else:
    print(f"File {DB_FILE} does not exist yet.")
