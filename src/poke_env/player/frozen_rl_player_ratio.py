# -*- coding: utf-8 -*-
"""
This module defines a frozen RL player
"""

from poke_env.player.player import Player
import tensorflow as tf
import numpy as np
import random


class FrozenRLPlayerRatio(Player):

    def __init__(self, trained_rl_model, model_name, *args, **kwargs):

        # create trained_rl_model attribute from input parameter
        self.trained_rl_model = trained_rl_model

        # specify model name - changes the way the best move is selected
        # since different models have different ways of choosing a best move
        self.model_name = model_name

        # inherit all attributes and methods from parent class
        Player.__init__(self, *args, **kwargs)
        
        
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

        return self.reward_computing_helper(battle, fainted_value=2, hp_value=1, victory_value=30)


    def _action_to_move(self, action, battle) -> str:

        """Converts actions to move orders.
            
        The conversion is done as follows:
        
        0 <= action < 4:
        The actionth available move in battle.available_moves is executed.
        4 <= action < 8:
        The action - 4th available move in battle.available_moves is executed, with
        z-move.
        8 <= action < 12:
        The action - 8th available move in battle.available_moves is executed, with
        mega-evolution.
        12 <= action < 18
        The action - 12th available switch in battle.available_switches is executed.
        
        If the proposed action is illegal, a random legal move is performed.
        
        :param action: The action to convert.
        :type action: int
        :param battle: The battle in which to act.
        :type battle: Battle
        :return: the order to send to the server.
        :rtype: str
        """
        if (
            action < 4
            and action < len(battle.available_moves)
            and not battle.force_switch
            ):
            return self.create_order(battle.available_moves[action])
        elif (
              not battle.force_switch
              and battle.can_z_move
              and 0 <= action - 4 < len(battle.active_pokemon.available_z_moves)
              ):
                return self.create_order(
                                         battle.active_pokemon.available_z_moves[action - 4], z_move=True
                                         )
        elif (
              battle.can_mega_evolve
              and 0 <= action - 8 < len(battle.available_moves)
              and not battle.force_switch
              ):
                return self.create_order(battle.available_moves[action - 8], mega=True)
        elif 0 <= action - 12 < len(battle.available_switches):
            return self.create_order(battle.available_switches[action - 12])
        else:
            return self.choose_random_move(battle)

            
    def choose_move(self, battle):
        r =random.random()
        if r<0.4:
            # If the player can attack, it will
            if battle.available_moves:

                # Finds the best move among available ones
                # Use trained rl model to select action

                if self.model_name == 'DQN' or self.model_name == 'SARSA':
                  # ONLY DQN & SARSA AGENTS SELECT ACTIONS LIKE THIS
                  battle_state = np.array([self.embed_battle(battle)])
                  battle_state = battle_state.flatten()
                  best_move = self.trained_rl_model.test_policy.select_action(battle_state)

                else:
                  best_move = self.trained_rl_model.select_action([self.embed_battle(battle)])

                return self._action_to_move(best_move, battle)
                #return best_move

            # If no attack is available, a random switch will be made
            else:
                return self.choose_random_move(battle)
        elif 0.4 <r <0.8:
            if battle.available_moves:
                # Finds the best move among available ones
                best_move = max(battle.available_moves, key=lambda move: move.base_power)
                return self.create_order(best_move)
                    
                    # If no attack is available, a random switch will be made
            else:
                return self.choose_random_move(battle)
        else:
            return self.choose_random_move(battle)


