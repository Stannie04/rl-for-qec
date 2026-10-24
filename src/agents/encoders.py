import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv

class CGNNEncoder(nn.Module):
    def __init__(self, config, env):
        super().__init__()

        hidden_dims = config.hidden_layers_cgnn
        feature_dim = env.code.feature_dim

        assert len(hidden_dims) > 0, "hidden_dims must contain at least one layer size."

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = feature_dim
        for out_dim in hidden_dims:
            self.convs.append(GraphConv(in_dim, out_dim))
            self.norms.append(nn.LayerNorm(out_dim))
            in_dim = out_dim

        self.name = "tanner"

    def forward(self, data):
        x = data.x

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            prev_x = x
            x = conv(x, data.edge_index)
            x = norm(x)
            x = F.relu(x)

            # Residual connection only if dimensions match
            if prev_x.shape[-1] == x.shape[-1]:
                x = x + prev_x

        return x


class MLP(nn.Module):
    def __init__(self, hidden_dim, hidden_dims, extra_dim=0):
        super().__init__()
        h1 = hidden_dims[0]
        self.src = nn.Linear(hidden_dim, h1, bias=False)
        self.dst = nn.Linear(hidden_dim, h1, bias=True)
        self.extra = nn.Linear(extra_dim, h1, bias=False) if extra_dim else None
        self.rest = nn.ModuleList(
            nn.Linear(hidden_dims[i - 1], hidden_dims[i]) for i in range(1, len(hidden_dims))
        )

    def node_terms(self, h_src, h_dst, extra_term=None):
        ps = self.src(h_src)
        if extra_term is not None:
            ps = ps + extra_term
        return ps, self.dst(h_dst)

    def edge_forward(self, ps_e, pd_e):
        z = ps_e + pd_e
        for lin in self.rest:          # ReLU between layers, none after the last
            z = lin(F.relu(z))         # (match this to your MLP class)
        return z


class NeuralBPEncoder(nn.Module):
    def __init__(self, config, env):
        super().__init__()
        H = config.neural_bp_hidden_dim
        mlp_hidden_dims = list(config.hidden_layers_mlp) + [H]
        self.H = H
        self.num_iters = config.neural_bp_iterations

        self.input_proj = nn.Linear(env.code.feature_dim, H)
        self.q_to_c = MLP(H, mlp_hidden_dims)
        self.c_to_q = MLP(H, mlp_hidden_dims, extra_dim=2)
        self.qubit_gru = nn.GRUCell(H, H)
        self.check_gru = nn.GRUCell(H, H)
        self.qubit_norm = nn.LayerNorm(H)
        self.check_norm = nn.LayerNorm(H)
        self.name = "nbp"
        self._ready = False

        self.set_graph(env.code.data.x, env.code.data.edge_index)  # Precompute graph topology
        self.to(config.device)  # Move the model to the specified device
        self.forward = torch.compile(self.forward, mode="reduce-overhead")  # Compile the forward method for performance

    @torch.no_grad()
    def set_graph(self, x0, edge_index):
        N, dev = x0.size(0), x0.device
        qmask = x0[:, 0] > 0.5
        q_idx = qmask.nonzero().squeeze(-1)
        c_idx = (~qmask).nonzero().squeeze(-1)
        Nq, Nc = q_idx.numel(), c_idx.numel()

        loc = torch.empty(N, dtype=torch.long, device=dev)   # global -> local index
        loc[q_idx] = torch.arange(Nq, device=dev)
        loc[c_idx] = torch.arange(Nc, device=dev)

        src, dst = edge_index.to(dev)

        def build(src_is_qubit, n_dst):
            sel = qmask[src] == src_is_qubit
            s, d = loc[src[sel]], loc[dst[sel]]
            order = torch.argsort(d, stable=True)             # sort by destination
            s, d = s[order], d[order]
            deg = torch.bincount(d, minlength=n_dst)
            uniform = int(deg[0]) if bool((deg == deg[0]).all()) else 0
            return s, d, uniform

        qc_s, qc_d, self.qc_deg = build(True, Nc)
        cq_s, cq_d, self.cq_deg = build(False, Nq)

        for name, t in dict(N=None, q_idx=q_idx, c_idx=c_idx, qc_s=qc_s, qc_d=qc_d,
                            cq_s=cq_s, cq_d=cq_d).items():
            if t is not None:
                self.register_buffer(name, t, persistent=False)
        self.N, self.Nq, self.Nc = N, Nq, Nc
        self._ready = True

    @staticmethod
    def _aggregate(msg, dst, n_dst, uniform_deg):
        B, E, H = msg.shape
        if uniform_deg:   # edges sorted by dst, same degree everywhere: no scatter
            return msg.reshape(B, n_dst, uniform_deg, H).sum(2)
        return msg.new_zeros(B, n_dst, H).index_add_(1, dst, msg)

    def forward(self, data):
        x = data.x
        flat = x.dim() == 2
        if flat:                                  # PyG-style [B*N, F] -> [B, N, F]
            x = x.view(-1, self.N, x.size(-1))
        B, H, Nq, Nc = x.size(0), self.H, self.Nq, self.Nc

        h = F.relu(self.input_proj(x))
        hq, hc = h[:, self.q_idx], h[:, self.c_idx]
        syn_term = self.c_to_q.extra(x[:, self.c_idx, 3:5])   # constant over iterations

        for _ in range(self.num_iters):
            # qubits -> checks
            ps, pd = self.q_to_c.node_terms(hq, hc)
            msg = self.q_to_c.edge_forward(ps[:, self.qc_s], pd[:, self.qc_d])
            agg = self._aggregate(msg, self.qc_d, Nc, self.qc_deg)
            c_new = self.check_gru(agg.reshape(-1, H), hc.reshape(-1, H)).view(B, Nc, H)
            hc = self.check_norm(c_new)

            # checks -> qubits (uses updated checks, as in your original)
            ps, pd = self.c_to_q.node_terms(hc, hq, syn_term)
            msg = self.c_to_q.edge_forward(ps[:, self.cq_s], pd[:, self.cq_d])
            agg = self._aggregate(msg, self.cq_d, Nq, self.cq_deg)
            q_new = self.qubit_gru(agg.reshape(-1, H), hq.reshape(-1, H)).view(B, Nq, H)
            hq = self.qubit_norm(q_new)

        out = h.index_copy(1, self.q_idx, hq).index_copy(1, self.c_idx, hc)
        return out.reshape(-1, H) if flat else out