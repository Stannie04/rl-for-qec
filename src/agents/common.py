import torch

def get_action_mask(state, env):
    # Based on encoding:
    # [is_qubit, is_x_syndrome, is_z_syndrome, x_syndrome, z_syndrome]
    # mask = H_x_T @ x_syndrome + H_z_T @ z_syndrome
    # Gets only qubits that are connected to active syndromes

    x_mask = env.code.H.x_T @ state.x[env.code.x_idx, 3]
    z_mask = env.code.H.z_T @ state.x[env.code.z_idx, 4]

    return (x_mask + z_mask) > 0


def get_qubit_mask(state):
    return torch.where(state.x[:,0] > 0.5)[0]


def _get_batch(data, num_nodes, device):
    if hasattr(data, "batch") and data.batch is not None:
        return data.batch
    return torch.zeros(num_nodes, dtype=torch.long, device=device)


def graph_mean_pool(x, data, batch):
    ptr = getattr(data, "ptr", None)

    # Single, unbatched graph.
    if ptr is None and getattr(data, "batch", None) is None:
        return x.mean(dim=0, keepdim=True)

    if ptr is not None:
        lengths = ptr[1:] - ptr[:-1]
        num_graphs = lengths.numel()
    else:
        num_graphs = data.num_graphs
        lengths = torch.bincount(
            batch, minlength=num_graphs
        )

    out = x.new_zeros((num_graphs, x.size(-1)))
    out.index_add_(0, batch, x)

    counts = lengths.to(dtype=x.dtype).unsqueeze(-1)
    return out / counts.clamp_min(1)


def scatter_add(x, index, dim_size):
    # Alternative to torch_scatter.scatter_add for SM_120 / torch 12.8 / Blackwell compatability
    out_shape = (dim_size,) + x.shape[1:]
    out = torch.zeros(out_shape, device=x.device, dtype=x.dtype)
    out.index_add_(0, index, x)
    return out