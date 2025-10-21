import numpy as np

class FrequencyAgent:
    def __init__(self, player_id):
        self.player_id = player_id
        self.opp_counts = [0, 0, 0]  # [Rock, Paper, Scissors]

    def reset(self):
        # Call this at the start of each new game
        self.opp_counts = [0, 0, 0]

    def select_action(self, obs, action_space):
        # Sometimes obs is just an empty/0D array at the start
        if not isinstance(obs, np.ndarray) or obs.shape == ():
            return action_space.sample()

        # Now obs should be a length-6 vector
        obs_array = obs
        opp_last_action = obs_array[3:]  # opponent’s past move one-hot

        if 1 in opp_last_action:
            opp_move = np.argmax(opp_last_action)
            self.opp_counts[opp_move] += 1

        # If no history yet, play random
        if sum(self.opp_counts) == 0:
            return action_space.sample()

        # Predict opponent's most frequent move
        predicted = np.argmax(self.opp_counts)

        # Counter it: (Rock=0, Paper=1, Scissors=2)
        best_move = (predicted + 1) % 3
        return best_move
