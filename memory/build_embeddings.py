import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")

df = pd.read_csv(os.path.join(PROJECT_ROOT, "dataset", "civicmind_episodes.csv"))

def create_memory_text(row):

    return f"""
    Weather: {row['weather']}
    Area Type: {row['area_type']}
    Event: {row['event_type']}
    Severity: {row['severity']}
    Construction Active: {row['construction_active']}
    Festival Active: {row['festival_active']}
    Peak Hour: {row['peak_hour']}
    Intervention: {row['intervention']}
    Citizens Affected: {row['citizens_affected']}
    Average Delay: {row['avg_delay_minutes']}
    Resolution Time: {row['resolution_time']}
    """

memory_texts = df.apply(create_memory_text,axis=1).tolist()


model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(
    memory_texts,
    show_progress_bar=True
)

print(embeddings.shape)

embeddings_array = np.array(
        embeddings,
        dtype = "float32"
)

dimension = embeddings_array.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings_array)

faiss.write_index(index, os.path.join(PROJECT_ROOT, "dataset", "civicmind_memory.index"))

df["memory_text"] = memory_texts

df.to_pickle(os.path.join(PROJECT_ROOT, "dataset", "episode_lookup.pkl"))

