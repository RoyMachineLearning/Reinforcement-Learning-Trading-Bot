# Overview

This project implements a Exchange Rate Trading Bot, trained using Deep Reinforcement Learning, specifically Deep Q-learning. Implementation is kept simple and as close as possible to the algorithm discussed in the paper, for learning purposes.

## Approach

This work uses a Model-free Reinforcement Learning technique called Deep Q-Learning (neural variant of Q-Learning).

At any given time (episode), an agent abserves it's current state (n-day window stock price representation), selects and performs an action (buy/sell), observes a subsequent state, receives some reward signal (difference in portfolio position) and lastly adjusts it's parameters based on the gradient of the loss computed.

There have been several improvements to the Q-learning algorithm over the years, and a few have been implemented in this project:

- [x] Vanilla DQN
- [x] DQN with fixed Q target distribution
- [x] Double DQN


Now you can open up a terminal and start training the agent:

# for Example DQN with Fixed Q Targets

python train.py data/EUR_USD_TRAIN.csv data/EUR_USD_VAL.csv --strategy t-dqn

Once you're done training, run the evaluation script and let the agent make trading decisions:

```bash
python eval.py data/EUR_USD_TEST.csv --model-name tdqn_EU_50 --debug
```
