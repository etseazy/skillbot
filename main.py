from pettingzoo.classic import tictactoe_v3, rps_v2
from agents.random_agent import RandomAgent
from agents.Minmax import MinimaxAgent
from agents.frequencyAgent import FrequencyAgent
from agents.dqn_agent import DQNAgent

def play_game(env, agent1, agent2):
    """
    Plays one full game in a PettingZoo environment between two agents.
    Returns: "player_1", "player_2", or "draw"
    """
    env.reset()
    env_agents = env.agents[:]
    rewards = {agent: 0 for agent in env_agents}

    # Map environment agent names to our standard ones
    agent_mapping = {
        env_agents[0]: agent1,
        env_agents[1]: agent2
    }
    
    # Reset agents if they have a reset() method
    for agent in agent_mapping.values():
        if hasattr(agent, "reset"):
            agent.reset()

    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()
        rewards[agent] += reward

        if termination or truncation:
            action = None
        else:
            action = agent_mapping[agent].select_action(obs, env.action_space(agent))
        env.step(action)

    # Decide winner based on total rewards
    p1_reward = rewards[env_agents[0]]
    p2_reward = rewards[env_agents[1]]
    if p1_reward > p2_reward:
        return "player_1"
    elif p2_reward > p1_reward:
        return "player_2"
    else:
        return "draw"


def run_tournament(env_fn, agent1, agent2, n_games=1000):
    """
    Runs multiple games and collects stats.
    """
    results = {"player_1": 0, "player_2": 0, "draw": 0}
    for _ in range(n_games):
        env = env_fn()
        outcome = play_game(env, agent1, agent2)
        results[outcome] += 1
    return results


# === Example usage ===
print("Choose a game to play:")
print("1. Tic-Tac-Toe")
print("2. Rock-Paper-Scissors")
choice = input("Enter 1 or 2: ").strip()

if choice == "1":
    print("\nChoose agents:")
    print("1. Minimax vs Random")
    print("2. DQN vs Random")
    print("3. DQN vs Minimax")
    agent_choice = input("Enter 1, 2, or 3: ").strip()
    
    if agent_choice == "1":
        agent1 = MinimaxAgent("player_1")
        agent2 = RandomAgent("player_2")
    elif agent_choice == "2":
        agent1 = DQNAgent("player_1", state_size=18, action_size=9)
        agent1.load("models/dqn_tictactoe.pth")
        agent1.epsilon = 0  # No exploration
        agent2 = RandomAgent("player_2")
    elif agent_choice == "3":
        agent1 = DQNAgent("player_1", state_size=18, action_size=9)
        agent1.load("models/dqn_tictactoe.pth")
        agent1.epsilon = 0
        agent2 = MinimaxAgent("player_2")
    else:
        print("Invalid choice")
        exit()
    
    results = run_tournament(
        tictactoe_v3.env,
        agent1,
        agent2,
        n_games=1000
    )
    
elif choice == "2":
    results = run_tournament(
        lambda: rps_v2.env(max_cycles=3),
        RandomAgent("player_1"),
        FrequencyAgent("player_2"),
        n_games=15000
    )
else:
    print("Invalid choice.")
    exit()

print("\n📊 Final results:", results)
total = sum(results.values())
print(f"Player 1 win rate: {results['player_1']/total*100:.1f}%")
print(f"Player 2 win rate: {results['player_2']/total*100:.1f}%")
print(f"Draw rate: {results['draw']/total*100:.1f}%")
