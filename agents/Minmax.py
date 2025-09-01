import numpy as np

class MinimaxAgent:
    def __init__(self, player_id):
        self.player_id = player_id
        # In PettingZoo: player_1 always goes first, player_2 second
        self.my_mark = 1 if player_id == "player_1" else -1
        self.opponent_mark = -self.my_mark

    def select_action(self, obs, action_space):
        mask = obs['action_mask']
        legal_moves = [i for i, v in enumerate(mask) if v]

        # reconstruct board from observation
        board = self.parse_board(obs["observation"])

        best_score = float("-inf")
        best_move = None

        for move in legal_moves:
            new_board = board.copy()
            new_board[move] = self.my_mark
            score = self.minimax(new_board, False)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def parse_board(self, obs_array):
        """Convert PettingZoo's (18,) obs into a flat (9,) board with -1,0,1 values"""
        my_positions = obs_array[:9]
        opp_positions = obs_array[9:]

        board = np.zeros(9, dtype=int)
        board[np.where(my_positions == 1)[0]] = self.my_mark
        board[np.where(opp_positions == 1)[0]] = self.opponent_mark

        return board

    def minimax(self, board, is_maximizing):
        result = self.check_winner(board)
        if result is not None:
            return result

        mask = self.get_action_mask(board)
        legal_moves = [i for i, v in enumerate(mask) if v]

        if is_maximizing:
            best = float("-inf")
            for move in legal_moves:
                new_board = board.copy()
                new_board[move] = self.my_mark
                val = self.minimax(new_board, False)
                best = max(best, val)
            return best
        else:
            best = float("inf")
            for move in legal_moves:
                new_board = board.copy()
                new_board[move] = self.opponent_mark
                val = self.minimax(new_board, True)
                best = min(best, val)
            return best

    def check_winner(self, board):
        # reshape board (9,) → (3,3)
        b = board.reshape(3, 3)

        # rows, cols, diagonals
        lines = list(b) + list(b.T) + [b.diagonal(), np.fliplr(b).diagonal()]

        for line in lines:
            if np.all(line == self.my_mark):
                return 1   # win
            if np.all(line == self.opponent_mark):
                return -1  # loss

        if np.all(board != 0):
            return 0  # draw

        return None  # game not finished

    def get_action_mask(self, board):
        return (board == 0).astype(int)
