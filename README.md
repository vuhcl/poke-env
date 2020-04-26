# The Pokemon Showdown Python environment

[![CircleCI](https://circleci.com/gh/hsahovic/poke-env.svg?style=svg)](https://circleci.com/gh/hsahovic/poke-env)
[![codecov](https://codecov.io/gh/hsahovic/poke-env/branch/master/graph/badge.svg)](https://codecov.io/gh/hsahovic/poke-env)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<a href="https://github.com/ambv/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![Documentation Status](https://readthedocs.org/projects/poke-env/badge/?version=latest)](https://poke-env.readthedocs.io/en/latest/?badge=latest)

A Python interface for training Reinforcement Learning bots to battle on [Pokemon Showdown](https://pokemonshowdown.com/), a fan-made simulation of Pokemon battling. It is maintained by the [Smogon Community] (https://www.smogon.com/) and support singles, doubles, and triples battles in all the games out so far (Generations 1 through 8).

This project aims at providing a Python environment for interacting in [Pokemon Showdown](https://pokemonshowdown.com/) battles, with reinforcement learning in mind. Currently, the agents in this project specifically follows the mechanics and rules in Generation 7.

## Keras RL Agents in Pokemon Showdown

This repo contains three different Keras RL agents from the keras-rl2 library that was used to train the Pokemon Showdown agents. The implementations and their directories are listed below:

+ SARSA in the `SARSAAgent` directory
+ Deep Q-Network (DQN) in the `DQNAgent` directory
+ Cross-entropy Method (CEM) in the `CEMAgent` directory

Each agent was trained for a set amount of steps - `NB_TRAINING_STEPS` - and evaluated against an agent for a set amount of episodes - `NB_EVALUATION_EPISODES`. Both parameters can be modified in the individual agent wrapper files, as well as the main wrapper notebook.

## Instructions for Training

+ Clone the repository and install all the requirements using `pip install -r requirements-clean.txt` We recommend that you set up a virtual environment. Instructions for setting up a virtual environment are linked [here](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).
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

## The Source Repository and Changes Made

This repository is forked from https://github.com/hsahovic/poke-env. Modifications made to the original repository include:

+ Functionality to train SARSA and CEM. The original repository used DQN and DDQN.
+ Functionality to save the results as `.csv` files and visualize the results, specifically the episode reward.
+ Opponent player class called `FrozenRLPlayer`, to complement the existing `RandomPlayer` and `MaxDamagePlayer`. The `FrozenRLPlayer` allows us to train our RL agent using self-play with previous iterations of the agent.
+ [Jupyter Notebook](https://github.com/nicolenair/poke-env/blob/master/src/rl_with_open_ai_gym_wrapper.ipynb) tutorial to enable a beginner to begin training Pokemon teams using our three RL agent types.

## Battling from Python

Here is a brief walkthrough of the files in [poke-env](https://github.com/nicolenair/poke-env/tree/master/src/poke_env):

+ The `data` folder contains `.json` files storing all the moves, Pokemon, as well as the type chart, with specific details. These files are loaded automatically with the code in the `data.py` file.
+ The `environment` folder contains the classes pertaining to the battle environment.
    + Most of the files define enumerations used to store constant values. These files are: `effect.py`, `field.py`, `move_category`, `pokemon_gender.py`, `pokemon_type.py`, `side_condition.py`, `status.py`, `weather.py`
    + The remaining files define object classes that represent the state of the battle and update this representation as a battle happens:
      + `battle.py` contains the code that defines the main class `Battle`, which represents the condition of the battle stage (weather, field, etc.) and the Pokemon teams of the two players. It also parses messages from the server to update this representation accordingly.
      + `pokemon.py` contains the code that defines the class `Pokemon`, which represents every individual Pokemon in the battling teams using the information from the Pokedex and requests sent from the `Battle` object.
      + `move.py` contains the code that defines the class `Move`, which represents a single move of an individual Pokemon. It gets the details of the move from the stored data and updates certain attributes when requested by the `Battle` object.
+ The `player` folder contains the various classes representing the player serving a range of purposes, from base classes to different opponents implementing different methods to choose a move. The player class will also communicate with the server to send requests and pass down updates to the `battle` class.

_Note:_ The representation referred to here is different from the state used when training and testing RL agents. Here, it is the full representation of the environment, including factors unobservable to either one or both of the two players. In contrast, the state representation in RL should be a much simpler one with only information available to the agent, and is specified in the `embed_battle()` method of the player class.

## Further Reading

For a walkthrough on understanding Pokemon Showdown as an RL environment and utilizing this repository to train RL agents to compete in the Pokemon Showdown environment, please refer to the following blog posts:

+ [Training a Pokemon Showdown agent using SARSA](https://medium.com/@vuhuychule/training-a-pok%C3%A9mon-battler-with-sarsa-algorithm-8ddee2c7732a)
+ [Training a Pokemon Showdown agent using DQN](https://medium.com/@hueyninglok/dqn-agent-for-pokemon-showdown-98169ccb50a3)
+ [Training a Pokemon Showdown agent using CEM](https://medium.com/@nicarina98/cross-entropy-method-for-training-a-pokemon-823d3590ae07)
+ [Improving the performance of Pokemon Showdown agents](https://medium.com/@vuhuychule/improving-the-performance-of-your-rl-model-without-learning-more-rl-253c1448182d)
