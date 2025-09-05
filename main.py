from pettingzoo.classic import tictactoe_v3, rps_v2
from agents.random_agent import RandomAgent
from agents.Minmax import MinimaxAgent

# Menu
print("Choose a game to play:")
print("1. Tic-Tac-Toe")
print("2. Rock-Paper-Scissors")
choice = input("Enter 1 or 2: ").strip()

if choice == "1":
    env = tictactoe_v3.env()
    agent1 = RandomAgent("player_1")
    agent2 = RandomAgent("player_2")
elif choice == "2":
    env = rps_v2.env(max_cycles=3)
    agent1 = RandomAgent("player_1")
    agent2 = RandomAgent("player_2")
else:
    print("Invalid choice.")
    exit()

agents = {"player_1": agent1, "player_2": agent2}

# track results
results = {"player_1": 0, "player_2": 0, "draw": 0}

# play many games
N_GAMES = 30000
for _ in range(N_GAMES):
    env.reset()
    # Get actual agent names from environment (handles different naming schemes)
    env_agents = env.agents[:]
    rewards = {agent: 0 for agent in env_agents}  # Track cumulative rewards
    
    # Map environment agent names to our agents
    agent_mapping = {}
    if len(env_agents) == 2:
        agent_mapping[env_agents[0]] = agents["player_1"]
        agent_mapping[env_agents[1]] = agents["player_2"]

    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()
        rewards[agent] += reward  # Accumulate rewards during the game
        # Do not break early; let all agents finish their final turn
        if termination or truncation:
            action = None
        else:
            action = agent_mapping[agent].select_action(obs, env.action_space(agent))
        env.step(action)

    # after the game ends, check the rewards to see who won
    # Map back to our standard player names
    p1_reward = rewards[env_agents[0]]
    p2_reward = rewards[env_agents[1]]
    if p1_reward > p2_reward:
        results["player_1"] += 1
    elif p2_reward > p1_reward:
        results["player_2"] += 1
    else:
        results["draw"] += 1

# final stats
print("After", N_GAMES, "games:")
print(results)
