# Scaling RL-driven Quantum Error Correction on Multivariate Bicycle Codes

This repository contains the code and resources for the formulation, construction and training of
reinforcement learning (RL)-driven quantum error correction (QEC) on multivariate bicycle codes.


```environments/```: Environment definition for the RL agent.

```agents/```: Directory containing RL agent implementations.

```docs/```: Documents containing changelog, roadmap and open issues.


## Running the code
To run the code, follow these steps:
### Cloning the repository

```bash
    git clone git@github.com:Stannie04/rl-for-qec.git
    cd rl-for-qec
  ```

### Installing dependencies
Make sure you have [Conda](https://docs.conda.io/en/latest/) installed. Then, create and activate the environment:
```bash
   conda env create -f environment.yml
   conda activate rl-for-qec
  ```

### Running the main training script
```bash
   python main.py
  ```