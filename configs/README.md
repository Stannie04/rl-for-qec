# The `configs/` directory

This directory contains configuration files for specifying various settings for training neural decoders on various QEC codes. The configuration is split into three main sections:

- `code_config.yml`: Specifies different QLDPC and toric codes, according to a specific naming convention. QLDPC codes can be constructed using:
  - `n`: The number of physical qubits.
  - `k`: The number of logical qubits.
  - `d`: The code distance.
  - `l` and `m`: The size of the lattice. The total number of physical qubits is given by `n = 2*l*m`.
  - `code_params`: Specifies the polynomials used to define the connectivity of the Tanner graph, in terms of matrices $A$ and $B$. $[[a, b, c], [d,e,f]]$ corresponds to $x^ay^bz^c + x^dy^ez^f$, following the construction of QLDPC codes noted in [Multivariate Bicycle Codes (Voss et al. 2025)](url:https://arxiv.org/abs/2406.19151). Toric code connectivity can be constructed using A = [[1,0,0], [0,0,0]] and B = [[0,1,0], [0,0,0]] (giving $A=x+1$ and $B=y+1$).

- `train_config.yml`: Specifies the hyperparameters for training the RL and supervised agents, as well as the router. The hyperparameters include training/evaluation timesteps, curriculum settings and parameters for Weights and Biases logging.
- `model_config.yml`: Specifies the architecture of the neural networks used for the RL and supervised agents, as well as the router. The architecture includes the number of layers, number of neurons per layer, activation functions, and other relevant parameters.