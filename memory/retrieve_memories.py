from build_embeddings import model
from build_embeddings import index
from build_embeddings import df
import numpy as np

query = """
    Heavy Rain
    Construction Active
    Peak Hour
    Traffic Congestion
"""

query_embeddings = model.encode(
        [query]
)

D, I = index.search(np.array(query_embeddings, dtype="float32"), k = 5)

for idx in I[0]:
    print(df.iloc[idx][ 
                [
                    "episode_id",
                    "event_type",
                    "weather",
                    "intervention"
                ]
            ]
    )
