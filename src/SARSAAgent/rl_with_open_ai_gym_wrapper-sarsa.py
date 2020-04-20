# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tensorflow as tf

from poke_env.player_configuration import PlayerConfiguration
from poke_env.player.env_player import Gen7EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer
from poke_env.server_configuration import LocalhostServerConfiguration

from rl.agents.sarsa import SARSAAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import sys
sys.path.append("..") # Adds higher directory to python modules path.

# We define our RL player
# It needs a state embedder and a reward computer, hence these two methods
class SimpleRLPlayer(Gen7EnvSinglePlayer):
    def embed_battle(self, battle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )

        # We count how many pokemons have not fainted in each team
        remaining_mon_team = (
            len([mon for mon in battle.team.values() if mon.fainted]) / 6
        )
        remaining_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        return np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [remaining_mon_team, remaining_mon_opponent],
            ]
        )

    def compute_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle, fainted_value=2, hp_value=1, victory_value=30
        )


class MaxDamagePlayer(RandomPlayer):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)

NB_TRAINING_STEPS = 100000
NB_EVALUATION_EPISODES = 100

# variable for naming .csv files.
# Change this according to whether the training process was carried out against a random player or a max damage player
TRAINING_OPPONENT = 'RandomPlayer'

tf.random.set_seed(0)
np.random.seed(0)

# This is the function that will be used to train the dqn
def dqn_training(player, dqn, nb_steps, filename):

    model = dqn.fit(player, nb_steps=nb_steps, visualize=False, verbose=2)

    # save model history to csv
    save_file = f"{filename}_trainlog_{nb_steps}eps.csv"
    print("===============================================")
    print(f"Saving model history as {save_file}")
    print("===============================================")
    pd.DataFrame(model.history).to_csv(save_file)

    player.complete_current_battle()


def dqn_evaluation(player, dqn, nb_episodes, filename):
    # Reset battle statistics
    player.reset_battles()
    model = dqn.test(player, nb_episodes=nb_episodes, visualize=False, verbose=False)

    # save model history to csv
    save_file = f"{filename}_testlog_{nb_episodes}eps.csv"
    print("===============================================")
    print(f"Saving model history as {save_file}")
    print("===============================================")
    pd.DataFrame(model.history).to_csv(save_file)

    print(
        "DQN Evaluation: %d victories out of %d episodes"
        % (player.n_won_battles, nb_episodes)
    )

if __name__ == "__main__":
    env_player = SimpleRLPlayer(
        player_configuration=PlayerConfiguration("satunicarina", "L.M.Montgomery7"),
        battle_format="gen7randombattle",
        server_configuration=LocalhostServerConfiguration,
    )

    opponent = RandomPlayer(
        player_configuration=PlayerConfiguration("duanicarina", "L.M.Montgomery7"),
        battle_format="gen7randombattle",
        server_configuration=LocalhostServerConfiguration,
    )

    second_opponent = MaxDamagePlayer(
        player_configuration=PlayerConfiguration("tiganicarina", "L.M.Montgomery7"),
        battle_format="gen7randombattle",
        server_configuration=LocalhostServerConfiguration,
    )

    # Output dimension
    n_action = len(env_player.action_space)

    model = Sequential()
    model.add(Dense(128, activation="elu", input_shape=(1, 10)))

    # Our embedding have shape (1, 10), which affects our hidden layer
    # dimension and output dimension
    # Flattening resolve potential issues that would arise otherwise
    model.add(Flatten())
    model.add(Dense(64, activation="elu"))
    model.add(Dense(n_action, activation="linear"))

    memory = SequentialMemory(limit=100000, window_length=1)

    # Epsilon greedy
    policy = EpsGreedyQPolicy(0.05)

    # Defining our DQN
    dqn = SARSAAgent(model=model, nb_actions=n_action, nb_steps_warmup=1000, policy=policy)

    dqn.compile(Adam(lr=0.0025), metrics=["mae"])

    # Training
    env_player.play_against(
        env_algorithm=dqn_training,
        opponent=opponent,
        env_algorithm_kwargs={"dqn": dqn, "nb_steps": NB_TRAINING_STEPS, "filename": TRAINING_OPPONENT},
    )
    model.save("model_%d" % NB_TRAINING_STEPS)

    # Evaluation
    print("Results against random player:")
    env_player.play_against(
        env_algorithm=dqn_evaluation,
        opponent=opponent,
        env_algorithm_kwargs={"dqn": dqn, "nb_episodes": NB_EVALUATION_EPISODES, "filename": f'({TRAINING_OPPONENT}_{NB_TRAINING_STEPS})RandomPlayer'},
    )

    print("\nResults against max player:")
    env_player.play_against(
        env_algorithm=dqn_evaluation,
        opponent=second_opponent,
        env_algorithm_kwargs={"dqn": dqn, "nb_episodes": NB_EVALUATION_EPISODES, "filename": f'({TRAINING_OPPONENT}_{NB_TRAINING_STEPS})MaxPlayer'},
    )
