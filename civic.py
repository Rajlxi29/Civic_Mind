import networkx as nx
from collections import defaultdict

G = nx.read_graphml("civicmind_graph.graphml")

print(f"No of edges: {G.number_of_edges()}")
print(f"No of nodes: {G.number_of_nodes()}")


edgetype = set()
for u, v, data in G.edges(data = True):
    if data.get("edge_type") == "PRECEDED":
        print(u," -> ",v, "time -> ", data.get("time_gap_hours"))

source_count = defaultdict(int)
transition_count = defaultdict(int)

for u, v, data in G.edges(data = True):
    
    source = G.nodes[u].get("event_type")
    target = G.nodes[v].get("event_type")

    if source and target:

        transition_count[
            (source, target)
        ] += 1

        source_count[source] += 1

confidence_score = {}
    
for (source, target), count in transition_count.items():
    confidence = (count / source_count[source])
    confidence_score[
        (source, target)
    ] = round(confidence, 3)


delay_store = defaultdict(list)

for u, v, data in G.edges(data = True):

    source = G.nodes[u].get("event_type")
    target = G.nodes[v].get("event_type")
    delay = data.get("time_gap_hours")

    if delay:
        delay_store[
            (source, target)
        ].append(float(delay))

avg_delay = {}

for edge, delays in delay_store.items():
    avg_delay[edge] = round(sum(delays)/len(delays), 2)

#Chains 
chain_count = defaultdict(int)

for middle in G.nodes():

    preds = list(G.predecessors(middle))
    succs = list(G.successors(middle))

    for p in preds:
        for s in succs:

            event1 = G.nodes[p].get("event_type")
            event2 = G.nodes[middle].get("event_type")
            event3 = G.nodes[s].get("event_type")

            if (
                event1 and
                event2 and
                event3
            ):

                chain = (
                    event1,
                    event2,
                    event3
                )

                chain_count[chain] += 1

#confidence
chain_confidence = {}
for chain, count in chain_count.items():

    first_event = chain[0]

    confidence = (
        count /
        source_count[first_event]
    )

    chain_confidence[
        chain
    ] = round(confidence, 3)

delay_chain = {}
for (a,b) in transition_count:

    for (c,d) in transition_count:

        if b == c:

            delay1 = avg_delay.get((a,b))
            delay2 = avg_delay.get((c,d))

            if delay1 is not None and delay2 is not None:

                chain = (a,b,d)

                delay_chain[chain] = round(
                    delay1 + delay2,
                    2
                )
                