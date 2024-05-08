import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

model = SentenceTransformer("paraphrase-MiniLM-L6-v2").to(device)

data = pd.read_csv("combined_files.csv")
video_descriptions = data["name"].tolist()

descriptions_embeddings = (
    model.encode(video_descriptions, convert_to_tensor=True).cpu().numpy()
)

np.save("embeddings.npy", descriptions_embeddings)

loaded_embeddings = np.load("embeddings.npy")

res = faiss.StandardGpuResources()

d = loaded_embeddings.shape[1]
gpu_index = faiss.GpuIndexFlatL2(res, d)
gpu_index.add(loaded_embeddings)

target_sentence = "xxxxxxx"
target_embedding = model.encode([target_sentence], convert_to_tensor=True).cpu().numpy()

k = 5
D, I = gpu_index.search(target_embedding, k)

print("Top similar video descriptions and their IDs:")
for i in range(k):
    idx = I[0][i]
    print(
        f"Video ID: {data.loc[idx, 'videoid']}, Description: {video_descriptions[idx]}, Distance: {D[0][i]}"
    )
