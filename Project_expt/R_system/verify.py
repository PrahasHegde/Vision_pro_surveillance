import pickle
import numpy as np

with open("embeddings.pkl", "rb") as f:
    embeddings, labels = pickle.load(f)

print("Number of embeddings:", len(embeddings))
print("Unique labels:", set(labels))
print("Example embedding shape:", np.array(embeddings[0]).shape)
