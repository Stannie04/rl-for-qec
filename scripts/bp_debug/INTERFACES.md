# BP debug: shared contract (issue #13)

Goal: find out why plain BP is very bad on the surface (`d7_surface`) and tile (`288_8_12_tile`) codes, using the
gross code (`144_12_12_ldpc`) as a working control. **Investigation only. Never edit anything under `src/`, `configs/`
or existing `tests/`.** Run everything from the repo root as `python -m scripts.bp_debug.<module>`.
Python: the default `python` (conda base; has ldpc 2.4.1, pymatching, torch). No new dependencies.

## Facts about the existing code (read these before coding)
- Config: `ConfigParser("configs", "bp", code_name, None)` (agent name `bp` exists in model_config.yml). `noise_model` is `"depolarizing"`.
- `QECEnv(config, shots)` (src/environment/env.py) holds `env.code` (QECCode, src/environment/code.py) with `H.x`, `H.z` (torch tensors), `n_data`, `logical`.
- Syndromes: `z_syndrome = H.z @ x_errors`, `x_syndrome = H.x @ z_errors` (mod 2). X errors are decoded with `H.z`, Z errors with `H.x`.
- Shots are arrays of shape `(N, 2, n)`: `[:,0,:]` = X error pattern, `[:,1,:]` = Z error pattern (a Y error is 1 in both). `n == n_data`.
- Repo decoders: `src/agents/classic.py` (`BPAgent`, `BPOSDAgent`, `BPLSDAgent`, `MWPMAgent`). `BPAgent` builds
  `ldpc.BpDecoder(H, error_rate=2*env.curriculum_error_rate/3, max_iter=30)` ONCE at construction; `env.curriculum_error_rate = curriculum_start_error_rate_144 * 144 / n_data`.
- Repo evaluation: `src/train_utils/inference.py::classical_inference(agent, env, state=None)` returns True on success; success means `is_syndrome_free() and not has_logical_error()`.
  `get_agent_and_inference(config, env, "bp")` returns `(agent, inference_fn)`. `env.reset()` serves shots from `env.shots` in order.
- Datasets: `src/train_utils/datasets.py`:
  - `create_dataset_from_uniform_shots(config, num_samples_per_error, max_error)`: weight w means **w X errors AND w Z errors at independent random positions** (so up to 2w qubits touched). Weight 5 on the 49-qubit code is a very dense error. Report this in every table caption.
  - `create_dataset_from_random_shots(config, num_samples, error_rate)`: iid depolarizing; each qubit has a Pauli with prob p, X/Y/Z uniform. Marginal prob of an X-type flip (X or Y) is `2p/3`.
- Logical check on the code object: `code.has_logical_error()` (requires zero syndrome).

## Output conventions
- Every script writes `results/bp_debug/<module_name>.json` (machine readable) and `results/bp_debug/<module_name>.md` (tables, with a one-line caption stating dataset, sample sizes, seeds).
- Every script has `main(argv=None)` with argparse flags `--codes` (default `d7_surface 288_8_12_tile 144_12_12_ldpc`), `--n-per-weight` (default 5000), `--seed` (default 0), and `--quick` (tiny sizes for tests, < 20 s).
- Seed numpy AND torch with `--seed`. Print progress with tqdm or plain prints.
- Shell commands time out after ~30 s in the card sandbox: run long runs in the background with a log file; `--quick` must finish in < 20 s on CPU.

## `scripts/bp_debug/common.py` (card BD1, everyone else imports it)
```python
def load_config(code_name: str, agent_name: str = "bp", noise_model: str = "depolarizing") -> ConfigParser
def make_env(config, shots=None) -> QECEnv
def uniform_shots(config, n_per_weight: int, max_weight: int = 5, seed: int = 0) -> np.ndarray   # wraps create_dataset_from_uniform_shots, seeded
def random_shots(config, n: int, p: float, seed: int = 0) -> np.ndarray                          # wraps create_dataset_from_random_shots, seeded
def single_qubit_shots(config, pauli: str) -> np.ndarray   # pauli in {"X","Y","Z"}; shape (n, 2, n): one shot per qubit
def syndromes(code, shots) -> tuple[np.ndarray, np.ndarray]  # (x_syndrome (N,m_x), z_syndrome (N,m_z)), uint8, computed with numpy from code.H, NOT with env.step
def independent_logical_failure(code, shots, x_pred, z_pred) -> np.ndarray  # bool (N,), see below
def decode_bp(code, shots, *, p_prior, max_iter=30, bp_method="product_sum", ms_scaling_factor=1.0, schedule="parallel", decoder="bp") -> dict
def weights(shots) -> np.ndarray   # nominal weight label per shot is NOT recoverable; this returns np.sum over (X|Z) support, i.e. number of qubits touched
```
- `independent_logical_failure`: residual x = shots[:,0]^x_pred, residual z = shots[:,1]^z_pred. Failure if residual syndrome (via `syndromes`) is non-zero, OR the residual anticommutes with any logical operator. Compute logical operators INDEPENDENTLY of `code.logical`: use `ldpc.mod2` / `galois` / numpy GF(2) linear algebra to find a basis of `ker(H_z) / rowspace(H_x)` (logical X) and `ker(H_x) / rowspace(H_z)` (logical Z), then test `L_z @ res_x` and `L_x @ res_z` mod 2. Cache the logical bases per code. Test it against `code.has_logical_error()` on random shots.
- `decode_bp(...)` returns `{"x_pred": (N,n) uint8, "z_pred": (N,n) uint8, "converged_x": (N,) bool, "converged_z": (N,) bool, "failure": (N,) bool}`.
  - `decoder` in {"bp", "bp_osd", "bp_lsd"}; maps to `ldpc.BpDecoder`, `ldpc.BpOsdDecoder`, `ldpc.BpLsdDecoder` (same args as `src/agents/classic.py`, plus the new keyword args; check the ldpc 2.4.1 signatures with `help()` and only pass args the class accepts).
  - X part: `ldpc decoder(H_z).decode(z_syndrome)`; Z part: `decoder(H_x).decode(x_syndrome)`.
  - `error_rate=p_prior` is passed through UNCHANGED (callers decide the 2p/3 convention).
  - `converged_*` from the decoder's `.converge` attribute (ldpc 2.x); for OSD/LSD report True when BP stage converged or post-processing ran.
  - `failure` uses `independent_logical_failure`.
- Include `tests/bp_debug/test_common.py` with `--quick`-sized fixtures: shapes/dtypes, `syndromes` equals `env.code` syndromes after `set_error_pattern`, `independent_logical_failure` agrees with `code.has_logical_error()` plus syndrome-free on 200 random low-weight shots for `d3_surface` and `d5_surface`, zero-error shots never fail, single-qubit X shot for d3 surface decoded by BP succeeds.

## Step modules (cards BD2..BD7). All use only `common.py` + repo code read-only.
- `step1_rerun.py`: for each code, uniform shots (weights 1..5) -> per-weight failure fraction for repo `bp`, `bp_osd`, `bp_lsd`, `mwpm` agents using `get_agent_and_inference` (the repo's own path, NOT `common.decode_bp`). Also the same via `common.decode_bp` with the repo's settings (`p_prior=2*curriculum_error_rate/3`, `max_iter=30`) and report agreement of the two pipelines per shot (count of disagreements). Table: code x weight x decoder.
- `step2_weight1.py`: `single_qubit_shots` for X, Y, Z; BP with repo settings. Table: code x Pauli -> failures / n, list failing qubit indices (first 20) and whether those qubits touch only 1 stabilizer of a type (boundary). Also run with `p_prior` in {0.001, 0.01, 0.1}.
- `step3_prior_sweep.py`: Part A weights 1..3 uniform shots; Part B iid depolarizing at p in {0.001..0.005} with `--n-per-weight` reinterpreted as shots per p (use max(n,20000) unless `--quick`). `p_prior` in {matched 2p/3 (per-shot p), 2p, p, repo value `2*curriculum/3`, 0.01, 0.05, 0.1}. Table: code x p x prior -> failure fraction.
- `step4_config_sweep.py`: one factor at a time around default (prior = 2p/3 at p=0.003, `max_iter` 30; iid shots): `max_iter` in {10,30,100,500}; `bp_method` in {product_sum, minimum_sum}; `ms_scaling_factor` in {0.5,0.625,0.75,0.9,1.0} (minimum_sum only); `schedule` in {parallel, serial}. Table: code x setting -> failure fraction, non-converged fraction.
- `step5_convergence_audit.py`: iid p=0.003 shots, repo-default BP. Partition failing shots into (a) not converged, (b) converged AND syndrome-free-but-logical, (c) converged, residual syndrome-free, `independent_logical_failure` False but repo `classical_inference` says failure (evaluation bug), (d) other. Counts and fractions per code, plus 3 example shots for each non-empty class (indices only).
- `step6_reference.py`: on `d7_surface` and `d5_surface`, iid p in {0.001, 0.003, 0.005} and uniform weights 1..3: pymatching MWPM (X via H_z, Z via H_x, `pymatching.Matching(H)`) failure rate with the independent check vs BP; also the repo `MWPMAgent` path. Report both, flag any disagreement between the independent logical check and repo `has_logical_error` (count + 3 example indices).

Each step module ends its `.md` with a short "Observation" section of at most 5 bullets stating what the numbers show (facts only; no speculation about fixes).
