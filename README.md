# Generalizable Quantum Error Correction using an Ensemble of Neural Decoders

This repository contains the code and resources for the formulation, construction and training of
reinforcement learning (RL)-driven quantum error correction (QEC) on multivariate bicycle codes, accompanying the thesis
"Generalizable Quantum Error Correction using an Ensemble of Neural Decoders" (link will be provided once available).


## Directory structure

The main directories in this repository are as follows. Read the README files in each directory for more details.

- ```src/```: All code used for creating, training and evaluating the RL-driven QEC framework.
- ```configs/```: YAML configuration files for the experiments, specifying the hyperparameters and settings for training and evaluation.
- ```results/```: Plots generated from the training and evaluation of the RL-driven QEC framework.
- ```environment.yml```: Conda environment file specifying the dependencies required to run the code.
- `main.py`: The main script for running experiments, training, evaluation, and analysis.
- `README.md`: This file.

When running the code, the following directories will be created:

- ```checkpoints/```: Saved model checkpoints during training.
- ```datasets/```: .npy files containing various sets of generated shots for training and evaluation.

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

### Running experiments
```bash
   python main.py -e <experiment> -a <agent> -c <code> [--verbose] [--run_name <run_name>]
  ```
`<experiment>`can be one of the following:
- `train_[rl|sl|router|all]`: for training the RL, SL, router or all models, respectively.
- `evaluate`: Evaluating the performance of the trained models on a test set, generating the corresponding results. </li>
- `analysis`: Analyzing various aspects of trained models and the code.
- `dataset`: Generate various datasets of shots for training and evaluation.
- `benchmark`: Evaluating the efficiency of performing inference on the environment.
- `render`: Visualizing the environment and the agent's actions.  

`<code>` specifies which code should be used, as defined in the `configs/` directory. The syntax for code names is "n_k_d_[ldpc|toric]".

`<agent>` only uses the "sac" option currently, but new agents can be added in the future.


## Reproducing the thesis

The experiments described in the thesis can be reproduced using
```
python main.py -e train_all -a sac -c 144_12_12_ldpc
```

followed by
```
python main.py -e evaluate -a sac -c 144_12_12_ldpc
```
Configuration files for all experiments are located in `configs/`.


## Citation
If you use this code in your research, please cite the following paper:
```
@thesis{ruijters2026ensembleqec,
  title={Generalizable Quantum Error Correction using an Ensemble of Neural Decoders},
  author={Ruijters, Stan},
  year={2026},
  school={Leiden University}
}
```

## Contact
For any questions or issues, please contact Stan Ruijters at [stan.ruijters@outlook.com](mailto:stan.ruiters@outlook.com).