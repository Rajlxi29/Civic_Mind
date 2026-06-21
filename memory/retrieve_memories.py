import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index(os.path.join(PROJECT_ROOT, "dataset", "civicmind_memory.index"))
df = pd.read_pickle(os.path.join(PROJECT_ROOT, "dataset", "episode_lookup.pkl"))

query = """
    Heavy Rain
    Construction Active
    Peak Hour
    Traffic Congestion
"""

query_embeddings = model.encode([query])

D, I = index.search(np.array(query_embeddings, dtype="float32"), k=5)

for idx in I[0]:
    print(df.iloc[idx][
        [
            "episode_id",
            "event_type",
            "weather",
            "intervention"
        ]
    ])
