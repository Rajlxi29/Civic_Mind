import os
import pandas as pd
import networkx as nx
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")

df = pd.read_csv(os.path.join(PROJECT_ROOT, "dataset", "civicmind_episodes.csv"))
df["timestamp"] = pd.to_datetime(df["timestamp"])

G = nx.DiGraph()

cause_cols = ["primary_cause", "secondary_cause", "tertiary_cause"]

concept_edge_count = defaultdict(int)

for _, row in df.iterrows():
    event_type = row["event_type"]

    if not G.has_node(event_type):
        G.add_node(event_type, node_type="concept")

    for col in cause_cols:
        cause = row[col]
        if pd.notna(cause):
            if not G.has_node(cause):
                G.add_node(cause, node_type="concept")
            concept_edge_count[(cause, event_type)] += 1

for (cause, event_type), weight in concept_edge_count.items():
    G.add_edge(cause, event_type, weight=weight, edge_type="CONCEPT")

for area, group in df.groupby("area_type"):
    group = group.sort_values("timestamp")
    prev = None
    for _, row in group.iterrows():
        ep_id = row["episode_id"]
        if not G.has_node(ep_id):
            G.add_node(
                ep_id,
                node_type="episode",
                event_type=row["event_type"],
                area_type=row["area_type"],
                weather=row["weather"],
                severity=row["severity"],
                timestamp=str(row["timestamp"]),
            )
        if prev is not None:
            gap_hours = (row["timestamp"] - prev["timestamp"]).total_seconds() / 3600
            if 0 < gap_hours <= 6:
                G.add_edge(
                    prev["episode_id"],
                    ep_id,
                    edge_type="PRECEDED",
                    time_gap_hours=round(gap_hours, 2),
                )
        prev = row

nx.write_graphml(G, os.path.join(PROJECT_ROOT, "dataset", "civicmind_graph.graphml"))

print(f"Nodes: {G.number_of_nodes()}")
print(f"  Episodes: {sum(1 for n in G.nodes if G.nodes[n].get('node_type') == 'episode')}")
print(f"  Concepts: {sum(1 for n in G.nodes if G.nodes[n].get('node_type') == 'concept')}")
print(f"Edges: {G.number_of_edges()}")
print(f"  PRECEDED: {sum(1 for _, _, d in G.edges(data=True) if d.get('edge_type') == 'PRECEDED')}")
print(f"  CONCEPT:  {sum(1 for _, _, d in G.edges(data=True) if d.get('edge_type') == 'CONCEPT')}")
