import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQNNetwork(nn.Module):
    """Neural network for Q-learning"""
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQNNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
    
    def forward(self, x):
        return self.network(x)


class DQNAgent:
    def __init__(self, player_id, state_size, action_size, learning_rate=0.001):
        self.player_id = player_id
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = 0.95          # discount factor
        self.epsilon = 1.0         # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.batch_size = 64
        
        # Neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQNNetwork(state_size, action_size).to(self.device)
        self.target_model = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        
        # Training stats
        self.training_stats = {
            'episode_rewards': [],
            'losses': [],
            'epsilon_history': []
        }
    
    def get_state_representation(self, obs):
        """Convert observation to flat numpy array"""
        if isinstance(obs, dict):
            if 'observation' in obs:
                state = obs['observation']
            else:
                state = np.array([obs.get('action_mask', [0] * self.action_size)])
        else:
            state = obs
        
        # Flatten to 1D array
        state = np.array(state).flatten()
        
        # Pad or truncate to expected state_size
        if len(state) < self.state_size:
            state = np.pad(state, (0, self.state_size - len(state)))
        elif len(state) > self.state_size:
            state = state[:self.state_size]
            
        return state
    
    def select_action(self, obs, action_space, training=False):
        """Select action using epsilon-greedy policy"""
        state = self.get_state_representation(obs)
        
        # Get legal moves
        if isinstance(obs, dict) and 'action_mask' in obs:
            legal_moves = [i for i, legal in enumerate(obs['action_mask']) if legal]
        else:
            legal_moves = list(range(action_space.n))
        
        if not legal_moves:
            return 0
        
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            return random.choice(legal_moves)
        
        # Exploitation: choose best legal action
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().numpy()[0]
        
        # Mask illegal actions
        masked_q_values = np.full(self.action_size, -np.inf)
        masked_q_values[legal_moves] = q_values[legal_moves]
        
        return np.argmax(masked_q_values)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def update_target_model(self):
        """Copy weights from model to target_model"""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def save(self, filepath):
        """Save model weights"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_stats': self.training_stats
        }, filepath)
    
    def load(self, filepath):
        """Load model weights"""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_stats = checkpoint['training_stats']