# RL_InvertedPendulum
Solving Inverted Pendulum with Reinforce Learning

## Implemented
- A basic system of inverted pendulum and training pipeline.
- Action value function in form of either discrete table or rbf kernel function.
- Sarsa & Q-Learning with/without TD-λ and eligibility traces.

## Examples
- run `python main.py` to train, it will save the results into `./output` or some other where determined by your config.
- run `python analysis.py` to draw trace during test. Please change path to the checkpoint in the code.

![image](https://github.com/volgachen/RL_InvertedPendulum/blob/master/images/trace.png)
