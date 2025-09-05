import numpy as np

class MinimaxAgent:
    def __init__(self, player_id):
        self.player_id = player_id
        # player_1 goes first → mark = 1
        self.my_mark = 1 if player_id == "player_1" else -1
        self.opponent_mark = -self.my_mark
        self.memo = {}  # Memoization cache

    def select_action(self, obs, action_space):
        mask = obs['action_mask']
        legal_moves = [i for i, v in enumerate(mask) if v]

        board = self.parse_board(obs["observation"])

        best_score = float("-inf")
        best_move = None

        for move in legal_moves:
            new_board = board.copy()
            new_board[move] = self.my_mark
            score = self.minimax(new_board, False, alpha=float("-inf"), beta=float("inf"))
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def parse_board(self, obs_array):
        """Convert PettingZoo's (3,3,2) obs into a flat (9,) board with -1,0,1 values"""
        # obs_array is (3,3,2) where [:,:,0] is current player, [:,:,1] is opponent
        my_positions = obs_array[:,:,0].flatten()
        opp_positions = obs_array[:,:,1].flatten()

        board = np.zeros(9, dtype=int)
        board[my_positions == 1] = self.my_mark
        board[opp_positions == 1] = self.opponent_mark

        return board

    def minimax(self, board, is_maximizing, alpha, beta):
        # Create a hashable key for memoization
        board_key = (tuple(board), is_maximizing)
        if board_key in self.memo:
            return self.memo[board_key]
            
        result = self.check_winner(board)
        if result is not None:
            self.memo[board_key] = result
            return result

        mask = self.get_action_mask(board)
        legal_moves = [i for i, v in enumerate(mask) if v]

        if is_maximizing:
            best = float("-inf")
            for move in legal_moves:
                new_board = board.copy()
                new_board[move] = self.my_mark
                val = self.minimax(new_board, False, alpha, beta)
                best = max(best, val)
                alpha = max(alpha, best)
                if beta <= alpha:  # prune
                    break
            self.memo[board_key] = best
            return best
        else:
            best = float("inf")
            for move in legal_moves:
                new_board = board.copy()
                new_board[move] = self.opponent_mark
                val = self.minimax(new_board, True, alpha, beta)
                best = min(best, val)
                beta = min(beta, best)
                if beta <= alpha:  # prune
                    break
            self.memo[board_key] = best
            return best

    def check_winner(self, board):
        b = board.reshape(3, 3)
        lines = list(b) + list(b.T) + [b.diagonal(), np.fliplr(b).diagonal()]

        for line in lines:
            if np.all(line == self.my_mark):
                return 1   # I win
            if np.all(line == self.opponent_mark):
                return -1  # Opponent wins

        if np.all(board != 0):
            return 0  # Draw

        return None  # Game still going

    def get_action_mask(self, board):
        return (board == 0).astype(int)
