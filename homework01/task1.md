# Task 1

## 1.1. Chess
S: finite set of states, describes possible board configurations

A: finite set of discrete actions, describes legal figure moves

R: negative reward for making a move, positive rewards for capturing figures, checks and the checkmate

P: deterministic, gives the next state given the current state and action (ie. the move)

## 1.2. LunarLander

S: continuous states, describes the coordinates and velocities of the lander and whether it is on the ground

A: finite set of discrete actions: do nothing, fire left orientation engine, fire main engine, fire right orientation
engine

R: negative reward for moving away from the landing pad or crashing, positive reward for coming to rest, extra positive
reward for landing on the landing pad


## 1.3. Model based RL

### Environment Dynamics

Environment dynamics describe how the environment changes given the current state and action. They can be stochastic
or deterministic.

Policy Evaluation algorithms evaluate the policy by calculating a state-value function V(s), which estimates the
expected return of each state. They use the environment dynamics to calculate the expected return (cumulative reward).
We do this so that we can find the optimal policy.

The Monte Carlo method estimates the expected return by sampling from the environment for _k_ times and averaging
over the returns.

Dynamic programming approach uses recursion to calculate the expected return of each state, since each subsequent
state depends on the previous state.

Policy Iteration algorithms iteratively evaluate and improve the policy. We may randomly initialize a policy, evaluate
it using a Policy Evaluation algorithm, and then improve it by choosing the best action for each state. We repeat this
until the policy converges (stops changing).

#### Reward Function

The reward function (R) is a function that maps the current state and action to a reward. It's usually probabilistic.
We usually denote it as `r(s, a)`.

In the chess example, the reward function could be `r(s, a) = -1` for every state and action, `r(s, a) = 1` if the
action captures a figure, and `r(s, a) = 100` if it's checkmate.

In the LunarLander example, the action of firing main engine gives -0.3 reward for every state, firing a side engine
gives -0.03 reward for every state, -100 reward if action caused a crash state, and so on.

#### State Transition Function

The state transition function (P) describes the environment dynamics. We can denote it as `p(s'|s, a)`, meaning the
probability of the next state, given the current state and action.

In the chess example, the state transition function could be `p(s'|s, a) = 1` for every state and action, since
chess is deterministic and we know how the environment will change for every board configuration and move.

In the FrozenLake example, slippery tiles are stochastic, so the state transition function could be
something like `p(s'|s, a) = [1/3, 1/3, 1/3]` if the agent is on a slippery tile and moves in any direction. This
would mean that the agent has a 1/3 chance of moving in the direction it wanted to, and 1/3 for each perpendicular
direction.

Together the reward function and the state transition function make up the environment dynamics and can be denoted
as `p(s', r|s, a)`.

#### Discussion

Are the environment dynamics generally known and can practically be used to solve a problem with RL?

Knowing the environment dynamics depends on the problem. For example, in chess, we know the rules of the game and
can calculate the environment dynamics. It is easy since it is deterministic. For the LunarLander, the state space is
continuous, so it is not easy to calculate the environment dynamics like in chess, but it is still possible since it
depends on a physics engine.

For more complicated problems, we may not know the environment dynamics so we cannot rely on them to improve the policy.