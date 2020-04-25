# The pokemon showdown Python environment

[![CircleCI](https://circleci.com/gh/hsahovic/poke-env.svg?style=svg)](https://circleci.com/gh/hsahovic/poke-env)
[![codecov](https://codecov.io/gh/hsahovic/poke-env/branch/master/graph/badge.svg)](https://codecov.io/gh/hsahovic/poke-env)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<a href="https://github.com/ambv/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![Documentation Status](https://readthedocs.org/projects/poke-env/badge/?version=latest)](https://poke-env.readthedocs.io/en/latest/?badge=latest)

A python interface for training Reinforcement Learning bots to battle on [pokemon showdown](https://pokemonshowdown.com/).

This project aims at providing a Python environment for interacting in [pokemon showdown](https://pokemonshowdown.com/) battles, with reinforcement learning in mind.

## Pokemon Showdown Environment
 Talk about poke-env briefly @vu
Cloning repo and running the env
Touch a lot on the original author’s stuff

## Keras RL Agents in Pokemon Showdown

This repo contains three different Keras RL agents from the keras-rl2 library that was used to train the Pokemon Showdown agents. The implementations and their directories are listed below:

+ SARSA in the `SARSAAgent` directory
+ Deep Q-Network (DQN) in the `DQNAgent` directory
+ Cross-entropy Method (CEM) in the `CEMAgent` directory

Each agent was trained for a set amount of steps - `NB_TRAINING_STEPS` - and evaluated against an agent for a set amount of episodes - `NB_EVALUATION_EPISODES`. Both parameters can be modified in the individual agent wrapper files, as well as the main wrapper notebook.

## Instructions for Training

+ Clone the repository and install all the requirements using ‘pip install requirements-clean.txt’ We recommend that you set up a virtual environment. Instructions for setting up a virtual environment are linked here.
+ You will also need to set up a Pokemon Showdown server to train your agents. This fork has been optimized for use with this poke-env repository: https://github.com/vuhcl/Pokemon-Showdown.git. Follow the installation instructions in the README to set up the server.
+ Two options:
  + run any of the three wrapper files from terminal 
    + files:
      + [dqn_open_ai_gym_wrapper.py](https://github.com/nicolenair/poke-env/blob/master/src/DQNAgent/dqn_open_ai_gym_wrapper.py)
      + [rl_with_open_ai_gym_wrapper-sarsa.py](https://github.com/nicolenair/poke-env/blob/master/src/SARSAAgent/rl_with_open_ai_gym_wrapper-sarsa.py)
      + [rl_with_open_ai_gym_wrapper-cem.py](https://github.com/nicolenair/poke-env/blob/master/src/CEMAgent/rl_with_open_ai_gym_wrapper-cem.py)
    + then use the visualization code associated with each wrapper to display the results:
      + [dqn_showdown_results.ipynb](https://github.com/nicolenair/poke-env/blob/master/src/DQNAgent/dqn_showdown_results.ipynb)
      + [SARSA-Showdown-Results.ipynb](https://github.com/nicolenair/poke-env/blob/master/src/SARSAAgent/SARSA-Showdown-Results.ipynb)
      + [CEM-Showdown-Results.ipynb](https://github.com/nicolenair/poke-env/blob/master/src/CEMAgent/CEM-Showdown-Results.ipynb)
  + follow along a simple tutorial for training & visualizing results for any of the three keras-rl2 models in this [jupyter notebook](https://github.com/nicolenair/poke-env/blob/master/src/rl_with_open_ai_gym_wrapper.ipynb)

## The Original Repository
This repository is forked from https://github.com/hsahovic/poke-env. Modifications made to the original repository include:

+ Functionality to train SARSA and CEM. The original repository used DQN and DDQN.
+ Functionality to save the results as `.csv` files and visualize the results, specifically the episode reward.
+ Opponent player class called `FrozenRLPlayer`, to complement the existing `RandomPlayer` and `MaxDamagePlayer`. The `FrozenRLPlayer` allows us to train our RL agent using self-play with previous iterations of the agent.
+ [Jupyter Notebook](https://github.com/nicolenair/poke-env/blob/master/src/rl_with_open_ai_gym_wrapper.ipynb) tutorial to enable a beginner to begin training Pokemon teams using our three RL agent types.

## Further Reading

For a walkthrough on understanding Pokemon Showdown as an RL environment and utilizing this repository to train RL agents to compete in the Pokemon Showdown environment, please refer to the following blog posts:

+ [Training a Pokemon Showdown agent using SARSA](https://medium.com/@vuhuychule/training-a-pok%C3%A9mon-battler-with-sarsa-algorithm-8ddee2c7732a)
+ [Training a Pokemon Showdown agent using DQN](https://medium.com/@hueyninglok/dqn-agent-for-pokemon-showdown-98169ccb50a3)
+ [Training a Pokemon Showdown agent using CEM](https://medium.com/@nicarina98/cross-entropy-method-for-training-a-pokemon-823d3590ae07)
