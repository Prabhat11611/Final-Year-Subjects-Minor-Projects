import numpy as np
from scipy.optimize import linear_sum_assignment

# Chess piece values
PIECE_VALUES = {
    'p': 1,  # Pawn
    'n': 3,  # Knight
    'b': 3,  # Bishop
    'r': 5,  # Rook
    'q': 9,  # Queen
    'k': 100 # King (high value to prioritize protection)
}

class ChessBoard:
    def __init__(self):
        self.board = [
            ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
            ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
            ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
        ]
        self.current_player = 'white'

    def print_board(self):
        for row in self.board:
            print(' '.join(row))
        print()

    def get_legal_moves(self, row, col):
        piece = self.board[row][col]
        moves = []

        if piece.lower() == 'p':  # Pawn
            direction = -1 if piece.isupper() else 1
            if 0 <= row + direction < 8 and self.board[row + direction][col] == '.':
                moves.append((row + direction, col))
            if (piece.isupper() and row == 6) or (piece.islower() and row == 1):
                if self.board[row + 2*direction][col] == '.':
                    moves.append((row + 2*direction, col))
            for dc in [-1, 1]:
                if 0 <= row + direction < 8 and 0 <= col + dc < 8:
                    target = self.board[row + direction][col + dc]
                    if target != '.' and target.isupper() != piece.isupper():
                        moves.append((row + direction, col + dc))

        # Add move generation for other pieces here (knight, bishop, rook, queen, king)
        # For simplicity, we'll only implement pawn moves in this example

        return moves

    def evaluate_board(self):
        score = 0
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece != '.':
                    value = PIECE_VALUES[piece.lower()]
                    if piece.isupper():
                        score += value
                    else:
                        score -= value
        return score

    def optimize_move(self):
        moves = []
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece != '.' and piece.isupper() == (self.current_player == 'white'):
                    legal_moves = self.get_legal_moves(row, col)
                    moves.extend([(row, col, *move) for move in legal_moves])

        if not moves:
            return None

        cost_matrix = np.zeros((len(moves), len(moves)))
        for i, move in enumerate(moves):
            # Make the move
            start_row, start_col, end_row, end_col = move
            piece = self.board[start_row][start_col]
            captured = self.board[end_row][end_col]
            self.board[end_row][end_col] = piece
            self.board[start_row][start_col] = '.'

            # Evaluate the board after the move
            score = self.evaluate_board()
            cost = -score if self.current_player == 'white' else score

            # Undo the move
            self.board[start_row][start_col] = piece
            self.board[end_row][end_col] = captured

            cost_matrix[i, i] = cost

        # Solve the linear assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        best_move_index = row_ind[0]

        return moves[best_move_index]

    def make_move(self, start_row, start_col, end_row, end_col):
        piece = self.board[start_row][start_col]
        self.board[end_row][end_col] = piece
        self.board[start_row][start_col] = '.'
        self.current_player = 'black' if self.current_player == 'white' else 'white'

def play_game():
    board = ChessBoard()
    for _ in range(50):  # Play for a maximum of 50 moves
        board.print_board()
        move = board.optimize_move()
        if move is None:
            print(f"Game over. {'Black' if board.current_player == 'white' else 'White'} wins!")
            break
        start_row, start_col, end_row, end_col = move
        board.make_move(start_row, start_col, end_row, end_col)
        print(f"{board.current_player.capitalize()}'s move: {chr(97+start_col)}{8-start_row} to {chr(97+end_col)}{8-end_row}")

if __name__ == "__main__":
    play_game()