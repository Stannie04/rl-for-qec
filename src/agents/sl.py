from torch import nn
import torch
from src.agents.common import get_qubit_mask
from src.agents.encoders import NeuralBPEncoder, CGNNEncoder

class SLAgent(nn.Module):
    def __init__(self, config, env, nbp_checkpoint=None, cgnn_checkpoint=None):
        super().__init__()
        self.config = config

        use_nbp = nbp_checkpoint is not None or (cgnn_checkpoint is None and config.use_neural_bp)
        out_dim = config.encoder_hidden_dim if use_nbp else config.hidden_layers_gnn[-1]

        self.encoder = NeuralBPEncoder(config, env) if use_nbp else CGNNEncoder(config, env)
        self.output_layer = nn.Linear(out_dim, 1)

        if nbp_checkpoint is not None:
            self.load_nbp(nbp_checkpoint)
        elif cgnn_checkpoint is not None:
            self.load_cgnn(cgnn_checkpoint)


    def forward(self, data):
        x = self.encoder(data)
        qubit_mask = get_qubit_mask(data)
        return torch.sigmoid(self.output_layer(x[qubit_mask])).squeeze(-1)  # Output shape: (num_qubits,)


    def save(self, path):
        torch.save({
        'encoder': self.encoder.state_dict(),
        'output_layer': self.output_layer.state_dict(),
        }, path)


    def load_nbp(self, path):
        checkpoint = torch.load(path, map_location=self.config.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.output_layer.load_state_dict(checkpoint['output_layer'])


    def load_cgnn(self, path):
        checkpoint = torch.load(path, map_location=self.config.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.output_layer.load_state_dict(checkpoint['output_layer'])