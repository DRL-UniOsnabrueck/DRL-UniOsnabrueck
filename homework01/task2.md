# Task 2.1 - Grid World Examples

Here are three grid world examples we found online:

1. FrozenLake: [https://www.gymlibrary.dev/environments/toy_text/frozen_lake/](https://www.gymlibrary.dev/environments/toy_text/frozen_lake/)
2. Cliff Walking: [https://www.gymlibrary.dev/environments/toy_text/cliff_walking/](https://www.gymlibrary.dev/environments/toy_text/cliff_walking/)
3. MiniGrid Empty: [https://minigrid.farama.org/environments/minigrid/EmptyEnv/](https://minigrid.farama.org/environments/minigrid/EmptyEnv/)

First two examples have the same 4 actions: up, down, left and right.

FrozenLake has slippery tiles that make the agent move in a random direction, so it is stochastic.

Cliff Walking is deterministic, but has a cliff that moves the agent back to the beginning with negative reward.

MiniGrid Empty has a triangle agent with 3 actions: move forward, turn left and turn right and is deterministic.