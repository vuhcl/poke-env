# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tensorflow as tf

import sys
sys.path.append("..") # Adds higher directory to python modules path.

# local python modules
from poke_env.player_configuration import PlayerConfiguration
from poke_env.player.env_player import Gen7EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer
from poke_env.server_configuration import LocalhostServerConfiguration

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


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

        # We count how many pokemons have fainted in each team
        fainted_mon_team = (
            len([mon for mon in battle.team.values() if mon.fainted]) / 6
        )
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        print()
        print(f"moves_base_power: {moves_base_power}")
        print(f"moves_dmg_multiplier: {moves_dmg_multiplier}")
        print(f"fainted_mon_team: {fainted_mon_team}")
        print(f"fainted_mon_opponent: {fainted_mon_opponent}")

        final_state_vector = np.concatenate(
                                [
                                    moves_base_power,
                                    moves_dmg_multiplier,
                                    [fainted_mon_team, fainted_mon_opponent],
                                ]
                            )

        print(f"Final state vector: {final_state_vector}")

        # Final vector with 10 components
        return final_state_vector


    def compute_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle, fainted_value=2, hp_value=1, victory_value=30
        )


NB_TRAINING_STEPS = 10
NB_EVALUATION_EPISODES = 0

# variable for naming .csv files. 
# Change this according to whether the training process was carried out against a random player or a max damage player
TRAINING_OPPONENT = 'RandomPlayer' 

tf.random.set_seed(0)
np.random.seed(0)

# This is the function that will be used to train the dqn
def dqn_training(player, dqn, nb_steps, filename):
    
    model = dqn.fit(player, nb_steps=nb_steps, visualize=False, verbose=2)

    player.complete_current_battle()



if __name__ == "__main__":
    env_player = SimpleRLPlayer(
        player_configuration=PlayerConfiguration("RandomPlayer1", "L.M.Montgomery7"),
        battle_format="gen7randombattle",
        server_configuration=LocalhostServerConfiguration,
    )

    random_opponent = RandomPlayer(
        player_configuration=PlayerConfiguration("RandomPlayer2", "L.M.Montgomery7"),
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

    memory = SequentialMemory(limit=10000, window_length=1)


    ############### print model deets ###############
    print(f"Model summary: {model.summary()}")

    inp = np.concatenate([
        [1.1, 0.8, 0., 1.3],
        [0.5, 0.5, 1., 1.], 
        [0.83333333, 0.83333333]
    ])

    inp = np.reshape(inp,(1,1,10))
    print(f"Example model input: {inp}")

    output = model.predict(inp)
    print(f"Model output: {output}")
    #################################################

    # Simple epsilon greedy
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0,
        nb_steps=10000,
    )

    # Defining our DQN
    dqn = DQNAgent(
        model=model,
        nb_actions=18,
        policy=policy,
        memory=memory,
        nb_steps_warmup=1,
        gamma=0.5,
        target_model_update=1,
        delta_clip=0.01,
        enable_double_dqn=True,
    )

    dqn.compile(Adam(lr=0.00025), metrics=["mae"])




    # Training
    env_player.play_against(
        env_algorithm=dqn_training,
        opponent=random_opponent,
        env_algorithm_kwargs={"dqn": dqn, "nb_steps": NB_TRAINING_STEPS, "filename": TRAINING_OPPONENT},
    )





