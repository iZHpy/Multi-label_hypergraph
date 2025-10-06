import torch
from torch_geometric.data import HeteroData
from collections import defaultdict
import math

class WeightedHypergraph:
    def __init__(self, num_labels):
        self.num_labels = num_labels
        self.hyperedges = defaultdict(lambda: {"weight": 0, "samples": set()})
        self.node_index = None
        self.edge_index = None
        self.edge_weight = None
        self.sample_to_edge = {}
        self.edge_to_id = {}
        self.id_to_edge = {}

    def create_from_labels(self, labels):
        for sample_id, sample_labels in enumerate(labels):
            if len(sample_labels) == 0:
                continue
            edge_key = tuple(sorted(sample_labels)) if len(sample_labels) > 1 else (sample_labels[0],)
            self.hyperedges[edge_key]["weight"] += 1
            self.hyperedges[edge_key]["samples"].add(sample_id)
            self.sample_to_edge[sample_id] = edge_key


        # Assign IDs to hyperedges
        for edge_id, edge_key in enumerate(self.hyperedges.keys()):
            self.edge_to_id[edge_key] = edge_id     # edge_id represents hyperedge ID
            self.id_to_edge[edge_id] = edge_key     # edge_key represents which labels are contained in each hyperedge
            
        # Create node_index, edge_index and edge_weight
        node_list = []
        edge_list = []
        weights = []
        node_to_id = {}

        for edge_key, edge_data in self.hyperedges.items():
            edge_id = self.edge_to_id[edge_key]
            for node in edge_key:
                # if node not in node_to_id:
                #     node_to_id[node] = len(node_to_id)
                node_list.append(node)
                edge_list.append(edge_id)
            log_weight = math.log(edge_data["weight"] + 1)  # prevent log(1)=0
            weights.append(log_weight)
        # self.node_index = torch.tensor(list(node_to_id.keys()), dtype=torch.long)   # node_index represents which label is contained in each node
        self.node_index = torch.arange(self.num_labels, dtype=torch.long)   # node_index represents which label is contained in each node
        self.edge_index = torch.tensor([node_list, edge_list], dtype=torch.long)    # edge_index shows which nodes are contained in each hypergraph
        self.edge_weight = torch.tensor(weights, dtype=torch.float)     # edge_weight represents the frequency of each hyperedge


        assert self.edge_index.shape[1] == len(node_list), "edge_index should contain all edge occurrences"

        assert len(self.edge_weight) == len(
            self.hyperedges), "edge_weight length should equal the number of unique edges"

        assert self.edge_index[1].max() < len(self.hyperedges), "Edge index out of bounds"



    def to_pytorch_geometric(self):
        data = HeteroData()

        # Add node type 'label'
        data['label'].x = self.node_index

        # Add node type 'hyperedge'
        data['hyperedge'].x = torch.arange(len(self.hyperedges))

        # Add edge type 'label-in-hyperedge'
        data['label', 'in', 'hyperedge'].edge_index = self.edge_index
        data['label', 'in', 'hyperedge'].edge_attr = self.edge_weight.unsqueeze(1)

        # Store additional metadata
        data.num_labels = self.num_labels
        data.num_hyperedges = len(self.hyperedges)

        return data

    def get_hyperedge_id(self, sample_id):
        if sample_id not in self.sample_to_edge:
            return None
        edge_key = self.sample_to_edge[sample_id]
        return self.edge_to_id[edge_key]

    def get_hyperedge_weight(self, edge_id):
        edge_key = self.id_to_edge[edge_id]
        return self.hyperedges[edge_key]["weight"]