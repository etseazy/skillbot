from pettingzoo.classic import tictactoe_v3
from agents.random_agent import RandomAgent
from agents.Minmax import MinimaxAgent

# make 2 bots
agent1 = RandomAgent("P1")
agent2 = MinimaxAgent("P2")
agents = {"player_1": agent1, "player_2": agent2}

# track results
results = {"player_1": 0, "player_2": 0, "draw": 0}

# play many games
N_GAMES = 100
for _ in range(N_GAMES):
    env = tictactoe_v3.env()
    env.reset()

    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()
        if termination or truncation:
            break
        action = agents[agent].select_action(obs, env.action_space(agent))
        env.step(action)

    # after the game ends, check the rewards to see who won
    p1_reward = env.rewards.get("player_1", 0)
    p2_reward = env.rewards.get("player_2", 0)
    if p1_reward > p2_reward:
        results["player_1"] += 1
    elif p2_reward > p1_reward:
        results["player_2"] += 1
    else:
        results["draw"] += 1

# final stats
print("After", N_GAMES, "games:")
print(results)
