from pettingzoo.classic import tictactoe_v3, rps_v2
from agents.dqn_agent import DQNAgent
from agents.random_agent import RandomAgent
import numpy as np
import matplotlib.pyplot as plt

def train_dqn_agent(game_env_fn, state_size, action_size, episodes=10000, 
                    opponent=None, save_path="models/dqn_agent.pth"):
    """
    Train a DQN agent through self-play or against an opponent
    """
    agent = DQNAgent("player_1", state_size, action_size)
    
    if opponent is None:
        # Self-play: create another DQN agent
        opponent = DQNAgent("player_2", state_size, action_size)
    
    episode_rewards = []
    win_rates = []
    losses = []
    
    print(f"🎮 Training DQN Agent for {episodes} episodes...")
    print(f"Device: {agent.device}")
    
    for episode in range(episodes):
        env = game_env_fn()
        env.reset()
        env_agents = env.agents[:]
        
        agent_mapping = {
            env_agents[0]: agent,
            env_agents[1]: opponent
        }
        
        episode_reward = 0
        episode_loss = []
        
        # Store experiences during the game
        experiences = []
        
        for env_agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            
            current_agent = agent_mapping[env_agent]
            
            # Only get state representation if this is the DQN agent
            if current_agent == agent:
                state = agent.get_state_representation(obs)
            
            if termination or truncation:
                action = None
                # Store final transition for DQN agent
                if experiences and current_agent == agent:
                    prev_state, prev_action = experiences[-1]
                    agent.remember(prev_state, prev_action, reward, state, True)
            else:
                # Select action based on agent type
                if current_agent == agent:
                    action = agent.select_action(obs, env.action_space(env_agent), training=True)
                    experiences.append((state, action))
                    episode_reward += reward
                else:
                    # Opponent (RandomAgent or other) just selects action normally
                    action = current_agent.select_action(obs, env.action_space(env_agent))
            
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
        
        # Evaluate win rate every 100 episodes
        if episode % 100 == 0 and episode > 0:
            win_rate = evaluate_agent(game_env_fn, agent, RandomAgent("player_2"), n_games=100)
            win_rates.append(win_rate)
            
            print(f"Episode {episode}/{episodes}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Win Rate vs Random: {win_rate:.1f}%")
            print(f"  Avg Reward: {np.mean(episode_rewards[-100:]):.3f}")
            if losses:
                print(f"  Avg Loss: {np.mean(losses[-100:]):.4f}")
            print()
        
        # Save checkpoint
        if episode % 1000 == 0 and episode > 0:
            agent.save(f"{save_path}.episode{episode}")
    
    # Save final model
    agent.save(save_path)
    print(f"✅ Training complete! Model saved to {save_path}")
    
    # Plot training curves
    plot_training_results(episode_rewards, win_rates, losses)
    
    return agent


def evaluate_agent(game_env_fn, agent, opponent, n_games=100):
    """Evaluate agent's win rate against an opponent"""
    wins = 0
    agent.epsilon = 0  # No exploration during evaluation
    
    for _ in range(n_games):
        env = game_env_fn()
        env.reset()
        env_agents = env.agents[:]
        
        agent_mapping = {
            env_agents[0]: agent,
            env_agents[1]: opponent
        }
        
        rewards = {env_agents[0]: 0, env_agents[1]: 0}
        
        for env_agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            rewards[env_agent] += reward
            
            if termination or truncation:
                action = None
            else:
                current_agent = agent_mapping[env_agent]
                # Check if agent has 'training' parameter
                if current_agent == agent:
                    action = current_agent.select_action(
                        obs, env.action_space(env_agent), training=False
                    )
                else:
                    action = current_agent.select_action(
                        obs, env.action_space(env_agent)
                    )
            env.step(action)
        
        if rewards[env_agents[0]] > rewards[env_agents[1]]:
            wins += 1
    
    return (wins / n_games) * 100


def plot_training_results(episode_rewards, win_rates, losses):
    """Visualize training progress"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Rewards
    axes[0].plot(episode_rewards, alpha=0.3)
    axes[0].plot(np.convolve(episode_rewards, np.ones(100)/100, mode='valid'))
    axes[0].set_title('Episode Rewards')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    
    # Win Rate
    if win_rates:
        axes[1].plot(win_rates)
        axes[1].set_title('Win Rate vs Random Agent')
        axes[1].set_xlabel('Evaluation (x100 episodes)')
        axes[1].set_ylabel('Win Rate (%)')
        axes[1].axhline(y=50, color='r', linestyle='--', label='Random baseline')
        axes[1].legend()
    
    # Loss
    if losses:
        axes[2].plot(losses, alpha=0.3)
        axes[2].plot(np.convolve(losses, np.ones(100)/100, mode='valid'))
        axes[2].set_title('Training Loss')
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    print("📊 Training plots saved to training_results.png")
    plt.show()


if __name__ == "__main__":
    print("Choose a game to train on:")
    print("1. Tic-Tac-Toe")
    print("2. Rock-Paper-Scissors")
    choice = input("Enter 1 or 2: ").strip()
    
    if choice == "1":
        # Tic-Tac-Toe: state = 3x3x2 = 18, actions = 9
        agent = train_dqn_agent(
            game_env_fn=tictactoe_v3.env,
            state_size=18,
            action_size=9,
            episodes=5000,
            opponent=RandomAgent("player_2"),
            save_path="models/dqn_tictactoe.pth"
        )
    elif choice == "2":
        # Rock-Paper-Scissors: state = 4, actions = 3
        agent = train_dqn_agent(
            game_env_fn=lambda: rps_v2.env(max_cycles=3),
            state_size=4,
            action_size=3,
            episodes=10000,
            opponent=RandomAgent("player_2"),
            save_path="models/dqn_rps.pth"
        )
    else:
        print("Invalid choice")