import numpy as np

class FrequencyAgent:
    def __init__(self, player_id):
        self.player_id = player_id
        self.opp_counts = [0, 0, 0]  # [Rock, Paper, Scissors]

    def select_action(self, obs, action_space):
        # Extract opponent's last move from observation
        print(obs)
        print(type(obs))
        obs_array = obs["observation"]
        opp_last_action = obs_array[3:]  # indices for opponent
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
