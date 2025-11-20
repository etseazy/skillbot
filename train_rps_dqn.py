from pettingzoo.classic import rps_v2
from agents.dqn_rps_agent import DQNRPSAgent
from agents.frequencyAgent import FrequencyAgent
from agents.random_agent import RandomAgent
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pygame')

def train_rps_dqn(episodes=15000, opponent_type="frequency"):
    """
    Train a DQN agent for Rock-Paper-Scissors with move history
    """
    agent = DQNRPSAgent("player_1", history_length=10)
    
    if opponent_type == "frequency":
        opponent = FrequencyAgent("player_2")
        print("🎯 Training against FrequencyAgent (has exploitable patterns)")
    else:
        opponent = RandomAgent("player_2")
        print("🎲 Training against RandomAgent (pure randomness)")
    
    episode_rewards = []
    win_rates_vs_freq = []
    win_rates_vs_random = []
    losses = []
    
    print(f"🎮 Training DQN Agent for {episodes} episodes...")
    print(f"Device: {agent.device}")
    print(f"State size: {agent.state_size} (includes move history)")
    print()
    
    for episode in range(episodes):
        env = rps_v2.env(max_cycles=3)
        env.reset()
        env_agents = env.agents[:]
        
        agent_mapping = {
            env_agents[0]: agent,
            env_agents[1]: opponent
        }
        
        # Reset agents
        agent.reset()
        if hasattr(opponent, 'reset'):
            opponent.reset()
        
        episode_reward = 0
        episode_loss = []
        experiences = []
        
        # Track moves for history updates
        moves = {env_agents[0]: [], env_agents[1]: []}
        
        for env_agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            
            current_agent = agent_mapping[env_agent]
            
            # Update opponent move history if we're the DQN agent
            if current_agent == agent and len(moves[env_agents[1]]) > len(agent.opponent_move_history):
                last_opp_move = moves[env_agents[1]][-1]
                agent.update_opponent_history(last_opp_move)
            
            if current_agent == agent:
                state = agent.get_state_representation(obs)
            
            if termination or truncation:
                action = None
                # Store final transition for DQN agent
                if experiences and current_agent == agent:
                    prev_state, prev_action = experiences[-1]
                    agent.remember(prev_state, prev_action, reward, state, True)
            else:
                if current_agent == agent:
                    action = agent.select_action(obs, env.action_space(env_agent), training=True)
                    experiences.append((state, action))
                    episode_reward += reward
                else:
                    # Opponent doesn't use 'training' parameter
                    action = current_agent.select_action(obs, env.action_space(env_agent))
                
                # Track the move
                moves[env_agent].append(action)
            
            env.step(action)
        
        # Train the agent
        if len(agent.memory) >= agent.batch_size:
            loss = agent.replay()
            if loss is not None:
                episode_loss.append(loss)
        
        # Update target network periodically
        if episode % 10 == 0:
            agent.update_target_model()
        
        # Track statistics
        episode_rewards.append(episode_reward)
        if episode_loss:
            losses.append(np.mean(episode_loss))
        
        # Evaluate every 100 episodes
        if episode % 100 == 0 and episode > 0:
            win_rate_freq = evaluate_rps_agent(agent, FrequencyAgent("player_2"), n_games=100)
            win_rate_random = evaluate_rps_agent(agent, RandomAgent("player_2"), n_games=100)
            win_rates_vs_freq.append(win_rate_freq)
            win_rates_vs_random.append(win_rate_random)
            
            print(f"Episode {episode}/{episodes}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Win Rate vs FrequencyAgent: {win_rate_freq:.1f}%")
            print(f"  Win Rate vs RandomAgent: {win_rate_random:.1f}%")
            print(f"  Avg Reward: {np.mean(episode_rewards[-100:]):.3f}")
            if losses:
                print(f"  Avg Loss: {np.mean(losses[-100:]):.4f}")
            print()
        
        # Save checkpoint
        if episode % 1000 == 0 and episode > 0:
            agent.save(f"models/dqn_rps_history.pth.episode{episode}")
    
    # Save final model
    agent.save("models/dqn_rps_history.pth")
    print(f"✅ Training complete! Model saved to models/dqn_rps_history.pth")
    
    # Plot training curves
    plot_rps_training(episode_rewards, win_rates_vs_freq, win_rates_vs_random, losses)
    
    return agent


def evaluate_rps_agent(agent, opponent, n_games=100):
    """Evaluate agent's win rate against an opponent"""
    original_epsilon = agent.epsilon
    agent.epsilon = 0  # No exploration during evaluation
    
    wins = 0
    
    for _ in range(n_games):
        env = rps_v2.env(max_cycles=3)
        env.reset()
        env_agents = env.agents[:]
        
        agent_mapping = {
            env_agents[0]: agent,
            env_agents[1]: opponent
        }
        
        # Reset agents
        agent.reset()
        if hasattr(opponent, 'reset'):
            opponent.reset()
        
        rewards = {env_agents[0]: 0, env_agents[1]: 0}
        moves = {env_agents[0]: [], env_agents[1]: []}
        
        for env_agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            rewards[env_agent] += reward
            
            current_agent = agent_mapping[env_agent]
            
            # Update opponent move history
            if current_agent == agent and len(moves[env_agents[1]]) > len(agent.opponent_move_history):
                last_opp_move = moves[env_agents[1]][-1]
                agent.update_opponent_history(last_opp_move)
            
            if termination or truncation:
                action = None
            else:
                if current_agent == agent:
                    action = current_agent.select_action(obs, env.action_space(env_agent), training=False)
                else:
                    # Opponent doesn't use 'training' parameter
                    action = current_agent.select_action(obs, env.action_space(env_agent))
                moves[env_agent].append(action)
            
            env.step(action)
        
        if rewards[env_agents[0]] > rewards[env_agents[1]]:
            wins += 1
    
    agent.epsilon = original_epsilon
    return (wins / n_games) * 100


def plot_rps_training(episode_rewards, win_rates_freq, win_rates_random, losses):
    """Visualize training progress"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Rewards
    axes[0].plot(episode_rewards, alpha=0.3, label='Episode Reward')
    if len(episode_rewards) > 100:
        axes[0].plot(np.convolve(episode_rewards, np.ones(100)/100, mode='valid'), label='Moving Avg')
    axes[0].set_title('Episode Rewards')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Win Rates
    if win_rates_freq:
        x = np.arange(len(win_rates_freq)) * 100
        axes[1].plot(x, win_rates_freq, marker='o', label='vs FrequencyAgent')
        axes[1].plot(x, win_rates_random, marker='s', label='vs RandomAgent')
        axes[1].axhline(y=33.33, color='r', linestyle='--', label='Random Baseline', alpha=0.5)
        axes[1].axhline(y=50, color='g', linestyle='--', label='Target (50%)', alpha=0.5)
        axes[1].set_title('Win Rates Over Training')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Win Rate (%)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # Loss
    if losses:
        axes[2].plot(losses, alpha=0.3, label='Loss')
        if len(losses) > 100:
            axes[2].plot(np.convolve(losses, np.ones(100)/100, mode='valid'), label='Moving Avg')
        axes[2].set_title('Training Loss')
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Loss')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rps_training_results.png')
    print("📊 Training plots saved to rps_training_results.png")
    plt.show()


if __name__ == "__main__":
    print("Choose opponent type:")
    print("1. FrequencyAgent (recommended - has exploitable patterns)")
    print("2. RandomAgent (baseline - pure randomness)")
    choice = input("Enter 1 or 2: ").strip()
    
    opponent_type = "frequency" if choice == "1" else "random"
    agent = train_rps_dqn(episodes=15000, opponent_type=opponent_type)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print("\nFinal Evaluation:")
    
    # Test against both opponents
    freq_win_rate = evaluate_rps_agent(agent, FrequencyAgent("player_2"), n_games=1000)
    random_win_rate = evaluate_rps_agent(agent, RandomAgent("player_2"), n_games=1000)
    
    print(f"\nWin rate vs FrequencyAgent: {freq_win_rate:.1f}% (1000 games)")
    print(f"Win rate vs RandomAgent: {random_win_rate:.1f}% (1000 games)")
    print(f"\nModel saved to: models/dqn_rps_history.pth")
