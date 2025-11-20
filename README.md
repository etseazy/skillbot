# 🤖 SkillBot - Multi-Agent Game AI System

A comprehensive reinforcement learning project featuring multiple AI agents playing Tic-Tac-Toe and Rock-Paper-Scissors. This project demonstrates classical game theory algorithms and modern deep reinforcement learning techniques.

## 🌟 Features

### Agents Implemented
1. **RandomAgent** - Baseline random player
2. **MinimaxAgent** - Optimal Tic-Tac-Toe player with alpha-beta pruning & memoization
3. **FrequencyAgent** - Pattern-tracking RPS player that exploits opponent tendencies
4. **DQNAgent** - Deep Q-Network for Tic-Tac-Toe (95%+ win rate vs Random)
5. **DQNRPSAgent** - Enhanced DQN with move history tracking for Rock-Paper-Scissors

### Games Supported
- **Tic-Tac-Toe** (via PettingZoo `tictactoe_v3`)
- **Rock-Paper-Scissors** (via PettingZoo `rps_v2`)

### Key Optimizations
- **Alpha-Beta Pruning**: ~95% node reduction in game tree search
- **Memoization**: 200-1000x speedup for Minimax (enables 30k+ games)
- **Experience Replay**: Efficient neural network training for DQN
- **Move History Tracking**: Enhanced state representation for pattern detection

## 📊 Performance Benchmarks

| Agent | Opponent | Win Rate | Notes |
|-------|----------|----------|-------|
| Minimax | Random | 99.7% | Near-perfect play |
| DQN (TTT) | Random | 95%+ | Learned optimal strategy |
| DQN (TTT) | Minimax | 0% (100% draws) | Both play optimally |
| DQN (RPS basic) | FrequencyAgent | 35.7% | Poor - no pattern detection |
| DQN (RPS enhanced) | FrequencyAgent | ~45-51%* | Improved with history tracking |

*Currently training - see `train_rps_dqn.py` for live results

## 🚀 Quick Start

### Installation
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install pettingzoo torch matplotlib numpy pygame
```

### Running Games
```bash
# Interactive menu for agent selection
python main.py

# Options:
# 1. Tic-Tac-Toe
#    - Minimax vs Random
#    - DQN vs Random
#    - DQN vs Minimax
# 2. Rock-Paper-Scissors
#    - Random vs Frequency
#    - DQN (basic) vs Frequency
#    - DQN (with history) vs Frequency
```

### Training Agents
```bash
# Train DQN for Tic-Tac-Toe (5000 episodes)
python train_dqn.py

# Train enhanced DQN for RPS (15000 episodes)
python train_rps_dqn.py
# Choose opponent type:
# 1. FrequencyAgent (recommended - has exploitable patterns)
# 2. RandomAgent (baseline - pure randomness)
```

## 📁 Project Structure

```
skillbot/
├── main.py                      # Interactive game runner
├── train_dqn.py                 # DQN training for Tic-Tac-Toe
├── train_rps_dqn.py            # Enhanced DQN training for RPS
├── agents/
│   ├── base.py                 # Base agent class
│   ├── random_agent.py         # Random baseline
│   ├── Minmax.py               # Optimal TTT with pruning
│   ├── frequencyAgent.py       # Pattern-tracking RPS agent
│   ├── dqn_agent.py           # General DQN implementation
│   └── dqn_rps_agent.py       # RPS-specific DQN with history
├── games/
│   └── tic_tac_toe.py         # Game utilities
└── models/
    ├── dqn_tictactoe.pth      # Trained TTT model
    └── dqn_rps_history.pth    # Trained RPS model (with history)
```

## 🧠 Technical Deep Dive

### Minimax Algorithm
- **Alpha-Beta Pruning**: Cuts search space by ~95%
- **Memoization**: Caches board states for O(1) lookup
- **Custom Board Parsing**: Converts PettingZoo's (3,3,2) observation to flat board
- **Performance**: Can evaluate 30,000+ games in reasonable time

### Deep Q-Network (DQN)
**Architecture:**
```
Input Layer (state_size)
    ↓
Hidden Layer (128 units, ReLU)
    ↓
Hidden Layer (128 units, ReLU)
    ↓
Output Layer (action_size, Q-values)
```

**Key Components:**
- **Experience Replay**: Buffer of 10,000 (state, action, reward, next_state, done) tuples
- **Target Network**: Separate network updated every 10 episodes for stability
- **Epsilon-Greedy**: Exploration rate decays from 1.0 → 0.01 over training
- **Bellman Equation**: Q(s,a) ← r + γ * max Q(s',a')

### Enhanced RPS Agent
**State Representation:**
```python
state_size = 6 * history_length + 4  # 64 dimensions for history_length=10

# Composition:
base_obs = [4]                    # Current round observation
my_history = [3 * 10]            # My last 10 moves (one-hot)
opponent_history = [3 * 10]      # Opponent's last 10 moves (one-hot)
```

**Why History Matters:**
- FrequencyAgent tracks opponent patterns
- Standard DQN (state_size=4) can't see opponent patterns
- Enhanced DQN (state_size=64) includes temporal information
- Enables pattern detection and counter-exploitation

## 🎓 What I Learned

### Classical AI
- Game tree search algorithms (Minimax)
- Alpha-beta pruning optimization
- Memoization techniques for dynamic programming
- Optimal play in deterministic games

### Modern Deep RL
- Deep Q-Networks implementation from scratch
- Experience replay for sample efficiency
- Target networks for training stability
- Epsilon-greedy exploration strategy
- State representation design for different game types

### Engineering
- Modular agent architecture with base classes
- Flexible tournament system for agent comparison
- Handling different PettingZoo environment naming schemes
- Training visualization with Matplotlib
- Model checkpointing and evaluation loops

## 📈 Future Enhancements

### Planned Features
- [ ] **ELO Rating System**: Competitive ranking for all agents
- [ ] **Web Dashboard**: Streamlit/Plotly for live visualization
- [ ] **Explainable AI**: Visualize agent "thinking" (Q-values, attention maps)
- [ ] **Advanced DQN**: Double DQN, Dueling DQN, Prioritized Experience Replay
- [ ] **Self-Play Training**: Agents improve by playing against themselves
- [ ] **More Games**: Connect Four, Chess (simplified), Go (9x9)
- [ ] **Multi-Agent Coordination**: Team-based games

## 🔧 Technical Requirements

- Python 3.8+
- PyTorch 2.0+
- PettingZoo
- NumPy
- Matplotlib
- Pygame (for rendering)

## 📖 References

- [PettingZoo Documentation](https://pettingzoo.farama.org/)
- [Deep Q-Networks (DQN) Paper](https://arxiv.org/abs/1312.5602)
- [Alpha-Beta Pruning](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning)
- [Experience Replay](https://arxiv.org/abs/1511.05952)

## 📝 License

This project is for educational purposes. Feel free to use and modify!

---

**Author**: Built as a learning project to explore classical and modern AI techniques in game playing.

**Last Updated**: Training enhanced RPS agent with move history tracking
