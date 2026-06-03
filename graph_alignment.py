
import torch
import torch.nn as nn

TOPO_FEAT_DIM = 8


def extract_topology_features(data, include_advanced=True):

    from torch_geometric.utils import degree, to_undirected

    num_companies = data['company'].x.size(0)
    device = data['company'].x.device
    company_edge_key = ('company', 'related_to', 'company')

    if company_edge_key not in data.edge_types:
        print(f"[TopoFeat] 警告: 边类型 {company_edge_key} 不存在，返回零特征。")
        return torch.zeros(num_companies, TOPO_FEAT_DIM, device=device)

    edge_index = data[company_edge_key].edge_index
    undirected_ei = to_undirected(edge_index, num_nodes=num_companies)

    d = degree(undirected_ei[0], num_nodes=num_companies, dtype=torch.float)
    d_norm = d / (d.max() + 1e-8)
    degree_feat = d_norm.unsqueeze(1)                        # [N, 1]

    if not include_advanced:
        pad = torch.full((num_companies, TOPO_FEAT_DIM - 1), 0.5, device=device)
        return torch.cat([degree_feat, pad], dim=1)

    try:
        import networkx as nx
        edge_list = undirected_ei.t().cpu().numpy()
        G = nx.Graph()
        G.add_nodes_from(range(num_companies))
        G.add_edges_from(edge_list)
        cc_dict = nx.clustering(G)
        clustering_feat = torch.tensor(
            [cc_dict.get(i, 0.0) for i in range(num_companies)],
            device=device, dtype=torch.float
        ).unsqueeze(1)                                       # [N, 1]
    except Exception:
        clustering_feat = torch.zeros(num_companies, 1, device=device)

    try:
        import networkx as nx
        G2 = nx.Graph()
        G2.add_nodes_from(range(num_companies))
        G2.add_edges_from(undirected_ei.t().cpu().numpy().tolist())
        pr_dict = nx.pagerank(G2, max_iter=50)
        pr_feat = torch.tensor(
            [pr_dict.get(i, 0.0) for i in range(num_companies)],
            device=device, dtype=torch.float
        ).unsqueeze(1)
        pr_feat = pr_feat / (pr_feat.max() + 1e-8)          # [N, 1]
    except Exception:
        pr_feat = torch.zeros(num_companies, 1, device=device)

    dc_feat = (d / (num_companies - 1 + 1e-8)).unsqueeze(1)  # [N, 1]

    try:
        from torch_geometric.utils import to_scipy_sparse_matrix
        adj = to_scipy_sparse_matrix(undirected_ei, num_nodes=num_companies)
        adj_sq = adj.dot(adj)
        tri = torch.tensor(
            adj_sq.diagonal() / 2.0, device=device, dtype=torch.float
        ).unsqueeze(1)
        tri_norm = tri / (tri.max() + 1e-8)                  # [N, 1]
    except Exception:
        tri_norm = torch.zeros(num_companies, 1, device=device)

    topo_feat = torch.cat([
        degree_feat,   # dim 0
        clustering_feat,  # dim 1
        pr_feat,       # dim 2
        dc_feat,       # dim 3
        tri_norm,      # dim 4
    ], dim=1)                                                # [N, 20]

    current_dim = topo_feat.size(1)
    if current_dim < TOPO_FEAT_DIM:
        padding = torch.zeros(num_companies, TOPO_FEAT_DIM - current_dim, device=device)
        topo_feat = torch.cat([topo_feat, padding], dim=1)   # [N, 8]

    return topo_feat


class GraphStructureAlignment(nn.Module):


    def __init__(self):
        super().__init__()

    def forward(self, data_src, data_tgt, s_src, s_tgt):

        device = s_src.device
        return torch.tensor(0.0, device=device), {}
