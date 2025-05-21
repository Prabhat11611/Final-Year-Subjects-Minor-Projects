# Using Hungarian Algorithm

import streamlit as st
import numpy as np
from scipy.optimize import linear_sum_assignment
import pandas as pd
import time

# Chess piece values and Unicode representations
PIECE_VALUES = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 100}
UNICODE_PIECES = {
    'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟',
    'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙',
    '.': ' '
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
        self.move_history = []

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
            start_row, start_col, end_row, end_col = move
            piece = self.board[start_row][start_col]
            captured = self.board[end_row][end_col]
            self.board[end_row][end_col] = piece
            self.board[start_row][start_col] = '.'

            score = self.evaluate_board()
            cost = -score if self.current_player == 'white' else score

            self.board[start_row][start_col] = piece
            self.board[end_row][end_col] = captured

            cost_matrix[i, i] = cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        best_move_index = row_ind[0]

        return moves[best_move_index]

    def make_move(self, start_row, start_col, end_row, end_col):
        piece = self.board[start_row][start_col]
        captured = self.board[end_row][end_col]
        self.board[end_row][end_col] = piece
        self.board[start_row][start_col] = '.'
        self.move_history.append((f"{chr(97+start_col)}{8-start_row}", f"{chr(97+end_col)}{8-end_row}", piece, captured))
        self.current_player = 'black' if self.current_player == 'white' else 'white'

def create_board_df(board):
    return pd.DataFrame([[UNICODE_PIECES[piece] for piece in row] for row in board])

def main():
    st.set_page_config(page_title="Advanced Chess AI", layout="wide")

    # Apply custom CSS for dark theme
    st.markdown("""
    <style>
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stDataFrame {
        background-color: #2C2C2C;
    }
    input[type="text"] {
        background-color: #2C2C2C;
        color: #FFFFFF;
    }
    h1, h2, h3, h4, h5, h6, p {
        color: #FFFFFF;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("Advanced Chess AI")

    # Initialize the game state
    if 'board' not in st.session_state:
        st.session_state.board = ChessBoard()
        st.session_state.game_over = False
        st.session_state.selected_piece = None

    # Create two columns for the layout
    col1, col2 = st.columns([3, 1])

    with col1:
        # Display the current board state
        st.dataframe(create_board_df(st.session_state.board.board), 
                     height=400, use_container_width=True)

        # Player move input
        if not st.session_state.game_over and st.session_state.board.current_player == 'white':
            start_square = st.text_input("Enter the starting square (e.g., e2):", placeholder="e2")
            end_square = st.text_input("Enter the ending square (e.g., e4):", placeholder="e4")
            
            if st.button("Make Move"):
                if start_square and end_square:
                    start_col, start_row = ord(start_square[0]) - 97, 8 - int(start_square[1])
                    end_col, end_row = ord(end_square[0]) - 97, 8 - int(end_square[1])
                    
                    if 0 <= start_row < 8 and 0 <= start_col < 8 and 0 <= end_row < 8 and 0 <= end_col < 8:
                        piece = st.session_state.board.board[start_row][start_col]
                        if piece.isupper():  # Check if it's a white piece
                            st.session_state.board.make_move(start_row, start_col, end_row, end_col)
                            st.experimental_rerun()
                        else:
                            st.error("Invalid move. Please select a white piece.")
                    else:
                        st.error("Invalid input. Please enter valid square coordinates.")
                else:
                    st.error("Please enter both starting and ending squares.")

        # AI move
        if not st.session_state.game_over and st.session_state.board.current_player == 'black':
            st.write("AI is thinking...")
            time.sleep(1)  # Simulate AI thinking time
            move = st.session_state.board.optimize_move()
            if move is None:
                st.session_state.game_over = True
                st.write("Game over. White wins!")
            else:
                start_row, start_col, end_row, end_col = move
                st.session_state.board.make_move(start_row, start_col, end_row, end_col)
                st.success(f"AI moved: {chr(97+start_col)}{8-start_row} to {chr(97+end_col)}{8-end_row}")
                st.experimental_rerun()

    with col2:
        # Display move history
        st.subheader("Move History")
        history_df = pd.DataFrame(st.session_state.board.move_history, 
                                  columns=["From", "To", "Piece", "Captured"])
        st.dataframe(history_df, height=300, use_container_width=True)

        # Display game status
        st.subheader("Game Status")
        st.write(f"Current player: {st.session_state.board.current_player.capitalize()}")
        st.write(f"Board evaluation: {st.session_state.board.evaluate_board()}")

        # Reset game button
        if st.button("Reset Game"):
            st.session_state.board = ChessBoard()
            st.session_state.game_over = False
            st.session_state.selected_piece = None
            st.experimental_rerun()

if __name__ == "__main__":
    main()
