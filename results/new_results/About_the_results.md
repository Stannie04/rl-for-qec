# Discussion about the results in the "results/new_results" folder

The experiment mainly focuses on splitting the environment into two parts:
- A training environment, where the agent learns to decode simple errors as quickly as possible.
- A testing environment, where the agent is evaluated on its ability to sustain a noisy environment as long as possible.

Previously, the agent was both trained en evaluated on the current "testing" objective. However, due to the large stochastity of this task,
this casued the agent to learn both simple errors and very complex/unrealistic errors at the same time. Consequently, the agent 
always converged to a policy in which it essentially passed on its actions, since it was not able to learn from its actions in the environment very well.
This behaviour is reflected in `train_on_eval_env_ldpc.png", where convergence happens around the same value that a static agent is able to achieve.

By training on a simpler environment first, the agent learns the code structure much more effectively, and is able to accurately decode simple errors.
By understanding the code structure better, the agent is able to generalize much better to the testing environment.
The results of training on the simplified environment are shown in `simplified_train_env.png", for both an [[18,4,4]] LDPC code and an [[18,2,3]] toric code.

The training curves in `simplified_train_env.png" are a bit misleading, since it looks like the agent is not learning at all. 
However, it is important to note that the means of both LDPC and toric code agents are significantly higher than the static agent
(approx. 200 steps for the static agent, approx. 1500 for LDPC and 2500 for toric), showing that both agents are able to counteract most errors in the environment.
Since the agents are able to decode simple errors accurately, termination essentially becomes a waiting game for the environment
to generate errors in more qubits than usual. This actually shows how quickly the agents are able to learn, since they find good policies almost immediately
(Toric within 10k steps of training, LDPC within 200k).

The increased frequency of outliers in the toric code agent, as well as the higher return of these outliers, shows that performance is
dependent on the complexity of the code, with LDPC codes being more difficult to learn.

Since the main bottleneck for the agents currently is learning more complex errors, this simplified training environment
will be seen as the first step in a potential curriculum learning pipeline. When the agents are able to understand simple errors,
training can be continued on (a hybrid of) the testing environment to learn more complex errors. As a final step in this pipeline,
adversarial learning may be able to be added back into the pipeline, to further improve the agent's performance on the testing environment.