import pickle

PKL_PATH = 'models\\face_encodings.pickle' # Make sure this matches your file name

try:
    with open(PKL_PATH, 'rb') as f:
        data = pickle.load(f)
        
    print(f"Data Type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"Keys: {data.keys()}")
        # Check first item to see format
        if 'embeddings' in data and len(data['embeddings']) > 0:
            print(f"First embedding shape: {data['embeddings'][0].shape}")
            
    elif isinstance(data, list):
        print(f"List Length: {len(data)}")
        if len(data) > 0:
            print(f"First Item: {data[0]}")
            print(f"Type of first item: {type(data[0])}")
            
    else:
        print("Data is neither dict nor list.")

except Exception as e:
    print(f"Error loading pickle: {e}")