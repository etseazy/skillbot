# ttt_random.py
from pettingzoo.classic import tictactoe_v3
import numpy as np
import random
from collections import Counter
import math
import argparse
import time

def select_random_action(obs, action_space):
    # preferred: use action_mask if provided
    if isinstance(obs, dict):
        if 'action_mask' in obs:
            mask = obs['action_mask']
            legal = [i for i, v in enumerate(mask) if v]
            if legal:
                return random.choice(legal)
        # sometimes board is inside 'observation'
        if 'observation' in obs:
            board = np.array(obs['observation']).flatten()
            if board.size == 9:
                legal = [i for i, v in enumerate(board) if v == 0]
                if legal:
                    return random.choice(legal)

    # fallback: sample from action_space (Discrete)
    try:
        a = action_space.sample()
        # if sample returns array/np.int, convert to int
        if isinstance(a, (np.integer, int)):
            return int(a)
        return a
    except Exception:
        return 0  # ultimate fallback

def simulate_games(n_games=1000, render=False, show_first_n_render=3):
    results = Counter()
    for game_i in range(n_games):
        env = tictactoe_v3.env()
        env.reset()
        agents = env.agents[:]
        rewards = {a: 0 for a in agents}  # track cumulative rewards

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            rewards[agent] += reward  # collect reward
            if termination or truncation:
                action = None
            else:
                action = select_random_action(obs, env.action_space(agent))
            env.step(action)
            if render and game_i < show_first_n_render:
                env.render()

        # Decide result
        if rewards[agents[0]] > rewards[agents[1]]:
            results['player1_win'] += 1
        elif rewards[agents[1]] > rewards[agents[0]]:
            results['player2_win'] += 1
        else:
            results['draw'] += 1
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=1000, help='How many games to simulate')
    parser.add_argument('--render', action='store_true', help='Render the first few games (useful for debugging)')
    args = parser.parse_args()

    print(f"Running {args.games} random‑vs‑random Tic‑Tac‑Toe games...")
    res = simulate_games(n_games=args.games, render=args.render)
    total = sum(res.values())
    print(res)
    print(f"player1 wins: {res['player1_win']} ({res['player1_win']/total:.2%})")
    print(f"player2 wins: {res['player2_win']} ({res['player2_win']/total:.2%})")
    print(f"draws: {res['draw']} ({res['draw']/total:.2%})")
