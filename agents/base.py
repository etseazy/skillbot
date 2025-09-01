class Agent:
    def __init__(self, name="Agent"):
        self.name = name
    
    def select_action(self, observation, action_space):
        """Decide on an action given the current game state"""
        raise NotImplementedError("This method should be overridden by subclasses")
