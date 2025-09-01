import random
from agents.base import Agent

class RandomAgent(Agent):
    def __init__(self, name="RandomAgent"):
        super().__init__(name)

    def select_action(self, observation, action_space):
        if "action_mask" in observation:
            mask = observation["action_mask"]
            legal = [i for i, v in enumerate(mask) if v]
            if legal:
                return random.choice(legal)
        # fallback if no mask exists
        return action_space.sample()
