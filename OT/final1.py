import streamlit as st
import numpy as np
import pandas as pd
import time
from functools import lru_cache
from typing import Dict, List, Tuple, Set
import zlib

# Chess piece values and Unicode representations
PIECE_VALUES = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 100}
UNICODE_PIECES = {
    'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟',
    'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙',
    '.': ' '
}

class TranspositionTable:
    def __init__(self, size: int = 1000000):
        self.size = size
        self.table: Dict[int, Tuple[float, int, str]] = {}
    
    def store(self, board_hash: int, value: float, depth: int, flag: str):
        """Store position evaluation in transposition table"""
        self.table[board_hash % self.size] = (value, depth, flag)
    
    def lookup(self, board_hash: int, depth: int) -> Tuple[float, str]:
        """Lookup position in transposition table"""
        if board_hash % self.size in self.table:
            value, stored_depth, flag = self.table[board_hash % self.size]
            if stored_depth >= depth:
                return value, flag
        return None, None

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
        self.transposition_table = TranspositionTable()
        self.move_cache = {}
        self.position_cache = {}
        
    def get_board_hash(self) -> int:
        """Generate a unique hash for the current board position"""
        board_str = ''.join([''.join(row) for row in self.board])
        return zlib.adler32(board_str.encode())
    
    @lru_cache(maxsize=10000)
    def get_legal_moves_cached(self, row: int, col: int, piece: str) -> List[Tuple[int, int]]:
        """Cached version of legal move generation"""
        moves = []
        
        if piece.lower() == 'p':  # Pawn moves optimization
            direction = -1 if piece.isupper() else 1
            new_row = row + direction
            
            if 0 <= new_row < 8:
                # Forward move
                if self.board[new_row][col] == '.':
                    moves.append((new_row, col))
                    # Initial two-square move
                    if (piece.isupper() and row == 6) or (piece.islower() and row == 1):
                        if self.board[new_row + direction][col] == '.':
                            moves.append((new_row + direction, col))
                
                # Captures
                for dc in [-1, 1]:
                    new_col = col + dc
                    if 0 <= new_col < 8:
                        target = self.board[new_row][new_col]
                        if target != '.' and target.isupper() != piece.isupper():
                            moves.append((new_row, new_col))
        
        elif piece.lower() == 'n':  # Knight moves optimization
            knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                          (1, -2), (1, 2), (2, -1), (2, 1)]
            for dr, dc in knight_moves:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    target = self.board[new_row][new_col]
                    if target == '.' or target.isupper() != piece.isupper():
                        moves.append((new_row, new_col))
        
        elif piece.lower() in ['r', 'b', 'q']:  # Sliding pieces optimization
            directions = []
            if piece.lower() in ['r', 'q']:
                directions += [(0, 1), (1, 0), (0, -1), (-1, 0)]
            if piece.lower() in ['b', 'q']:
                directions += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
                
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                while 0 <= new_row < 8 and 0 <= new_col < 8:
                    target = self.board[new_row][new_col]
                    if target == '.':
                        moves.append((new_row, new_col))
                    elif target.isupper() != piece.isupper():
                        moves.append((new_row, new_col))
                        break
                    else:
                        break
                    new_row, new_col = new_row + dr, new_col + dc
        
        elif piece.lower() == 'k':  # King moves optimization
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    new_row, new_col = row + dr, col + dc
                    if 0 <= new_row < 8 and 0 <= new_col < 8:
                        target = self.board[new_row][new_col]
                        if target == '.' or target.isupper() != piece.isupper():
                            moves.append((new_row, new_col))
        
        return moves

    @lru_cache(maxsize=1000)
    def evaluate_position_cached(self, board_hash: int) -> float:
        """Cached position evaluation"""
        if board_hash in self.position_cache:
            return self.position_cache[board_hash]
        
        score = 0
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece != '.':
                    value = PIECE_VALUES[piece.lower()]
                    multiplier = 1 if piece.isupper() else -1
                    score += value * multiplier
                    
                    # Position-based bonuses
                    if piece.lower() in ['p', 'n', 'b']:
                        center_distance = (3.5 - abs(col - 3.5) - abs(row - 3.5)) * 0.1
                        score += center_distance * multiplier
        
        self.position_cache[board_hash] = score
        return score

    def minimax_with_memory(self, depth: int, alpha: float, beta: float, 
                           maximizing_player: bool) -> float:
        """Minimax algorithm with alpha-beta pruning and transposition table"""
        board_hash = self.get_board_hash()
        
        # Transposition table lookup
        value, flag = self.transposition_table.lookup(board_hash, depth)
        if value is not None:
            return value
        
        if depth == 0:
            value = self.evaluate_position_cached(board_hash)
            self.transposition_table.store(board_hash, value, depth, 'exact')
            return value
        
        moves = self.get_all_moves()
        if maximizing_player:
            max_eval = float('-inf')
            for move in moves:
                start_row, start_col, end_row, end_col = move
                captured = self.board[end_row][end_col]
                self.make_move(start_row, start_col, end_row, end_col)
                eval = self.minimax_with_memory(depth - 1, alpha, beta, False)
                self.undo_move(start_row, start_col, end_row, end_col, captured)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            self.transposition_table.store(board_hash, max_eval, depth, 
                                         'exact' if alpha < beta else 'beta')
            return max_eval
        else:
            min_eval = float('inf')
            for move in moves:
                start_row, start_col, end_row, end_col = move
                captured = self.board[end_row][end_col]
                self.make_move(start_row, start_col, end_row, end_col)
                eval = self.minimax_with_memory(depth - 1, alpha, beta, True)
                self.undo_move(start_row, start_col, end_row, end_col, captured)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            self.transposition_table.store(board_hash, min_eval, depth, 
                                         'exact' if alpha < beta else 'alpha')
            return min_eval

    def get_all_moves(self) -> List[Tuple[int, int, int, int]]:
        """Generate all legal moves for current player"""
        moves = []
        board_hash = self.get_board_hash()
        
        if board_hash in self.move_cache:
            return self.move_cache[board_hash]
        
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece != '.' and piece.isupper() == (self.current_player == 'white'):
                    legal_moves = self.get_legal_moves_cached(row, col, piece)
                    moves.extend([(row, col, *move) for move in legal_moves])
        
        self.move_cache[board_hash] = moves
        return moves

    def get_best_move(self, depth: int) -> Tuple[int, int, int, int]:
        """Find the best move using minimax with optimizations"""
        best_move = None
        best_eval = float('-inf') if self.current_player == 'white' else float('inf')
        alpha = float('-inf')
        beta = float('inf')
        moves = self.get_all_moves()

        for move in moves:
            start_row, start_col, end_row, end_col = move
            captured = self.board[end_row][end_col]
            self.make_move(start_row, start_col, end_row, end_col)
            eval = self.minimax_with_memory(depth - 1, alpha, beta, 
                                          self.current_player == 'black')
            self.undo_move(start_row, start_col, end_row, end_col, captured)

            if self.current_player == 'white':
                if eval > best_eval:
                    best_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
            else:
                if eval < best_eval:
                    best_eval = eval
                    best_move = move
                beta = min(beta, eval)

            if beta <= alpha:
                break

        return best_move

    def make_move(self, start_row: int, start_col: int, end_row: int, end_col: int):
        """Make a move on the board"""
        piece = self.board[start_row][start_col]
        captured = self.board[end_row][end_col]
        self.board[end_row][end_col] = piece
        self.board[start_row][start_col] = '.'
        self.move_history.append((
            f"{chr(97+start_col)}{8-start_row}",
            f"{chr(97+end_col)}{8-end_row}",
            piece,
            captured
        ))
        self.current_player = 'black' if self.current_player == 'white' else 'white'
        
        # Clear relevant caches
        self.get_legal_moves_cached.cache_clear()
        self.evaluate_position_cached.cache_clear()

    def undo_move(self, start_row: int, start_col: int, end_row: int, end_col: int, 
                  captured: str):
        """Undo a move on the board"""
        piece = self.board[end_row][end_col]
        self.board[start_row][start_col] = piece
        self.board[end_row][end_col] = captured
        self.current_player = 'black' if self.current_player == 'white' else 'white'
        self.move_history.pop()
        
        # Clear relevant caches
        self.get_legal_moves_cached.cache_clear()
        self.evaluate_position_cached.cache_clear()

def create_board_df(board):
    """Create a DataFrame representation of the board"""
    return pd.DataFrame([[UNICODE_PIECES[piece] for piece in row] for row in board])



def main():
    st.set_page_config(page_title="Optimized Chess AI", layout="wide")

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
    table {
        color: white;
    }
    .move-history {
        margin-top: 20px;
        padding: 10px;
        background-color: #2C2C2C;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("Optimized Chess AI")

    # Initialize the game state
    if 'board' not in st.session_state:
        st.session_state.board = ChessBoard()
        st.session_state.game_over = False
        st.session_state.selected_piece = None
        st.session_state.difficulty = 3  # Default difficulty level

    # Create sidebar for settings
    with st.sidebar:
        st.header("Game Settings")
        difficulty = st.slider("AI Difficulty (Search Depth)", 
                             min_value=1, 
                             max_value=5, 
                             value=st.session_state.difficulty,
                             help="Higher values make the AI stronger but slower")
        
        if difficulty != st.session_state.difficulty:
            st.session_state.difficulty = difficulty
            
        # Add performance metrics
        if 'performance' not in st.session_state:
            st.session_state.performance = {
                'positions_evaluated': 0,
                'cache_hits': 0,
                'avg_move_time': 0
            }
        
        st.header("Performance Metrics")
        st.write(f"Positions evaluated: {st.session_state.performance['positions_evaluated']}")
        st.write(f"Cache hits: {st.session_state.performance['cache_hits']}")
        st.write(f"Average move time: {st.session_state.performance['avg_move_time']:.2f}s")

    # Create two columns for the layout
    col1, col2 = st.columns([3, 1])

    with col1:
        # Display the current board state
        st.dataframe(create_board_df(st.session_state.board.board), 
                     height=400, 
                     use_container_width=True)

        # Player move input
        if not st.session_state.game_over and st.session_state.board.current_player == 'white':
            move_col1, move_col2 = st.columns(2)
            
            with move_col1:
                start_square = st.text_input("Starting square (e.g., e2):", 
                                           placeholder="e2",
                                           key="start_square")
            
            with move_col2:
                end_square = st.text_input("Ending square (e.g., e4):", 
                                         placeholder="e4",
                                         key="end_square")
            
            if st.button("Make Move", key="make_move"):
                if start_square and end_square:
                    try:
                        start_col = ord(start_square[0].lower()) - 97
                        start_row = 8 - int(start_square[1])
                        end_col = ord(end_square[0].lower()) - 97
                        end_row = 8 - int(end_square[1])
                        
                        if (0 <= start_row < 8 and 0 <= start_col < 8 and 
                            0 <= end_row < 8 and 0 <= end_col < 8):
                            
                            piece = st.session_state.board.board[start_row][start_col]
                            if piece.isupper():  # Check if it's a white piece
                                # Validate move
                                legal_moves = st.session_state.board.get_legal_moves_cached(
                                    start_row, start_col, piece)
                                if (end_row, end_col) in legal_moves:
                                    st.session_state.board.make_move(
                                        start_row, start_col, end_row, end_col)
                                    st.experimental_rerun()
                                else:
                                    st.error("Invalid move. This piece cannot move there.")
                            else:
                                st.error("Invalid move. Please select a white piece.")
                        else:
                            st.error("Invalid input. Please enter valid square coordinates.")
                    except (IndexError, ValueError):
                        st.error("Invalid input format. Please use format 'e2' for squares.")
                else:
                    st.error("Please enter both starting and ending squares.")

        # AI move
        if not st.session_state.game_over and st.session_state.board.current_player == 'black':
            st.write("AI is thinking...")
            start_time = time.time()
            
            # Get AI move with current difficulty setting
            move = st.session_state.board.get_best_move(depth=st.session_state.difficulty)
            
            end_time = time.time()
            move_time = end_time - start_time
            
            # Update performance metrics
            st.session_state.performance['avg_move_time'] = (
                st.session_state.performance['avg_move_time'] + move_time) / 2
            
            if move is None:
                st.session_state.game_over = True
                st.write("Game over. White wins!")
            else:
                start_row, start_col, end_row, end_col = move
                st.session_state.board.make_move(start_row, start_col, end_row, end_col)
                
                move_san = f"{chr(97+start_col)}{8-start_row} to {chr(97+end_col)}{8-end_row}"
                st.success(f"AI moved: {move_san} (took {move_time:.1f}s)")
                
                # Check for checkmate or stalemate
                if not st.session_state.board.get_all_moves():
                    st.session_state.game_over = True
                    st.write("Game over. Black wins!")
                
                st.experimental_rerun()

    with col2:
        # Display move history
        st.subheader("Move History")
        if st.session_state.board.move_history:
            history_df = pd.DataFrame(
                st.session_state.board.move_history,
                columns=["From", "To", "Piece", "Captured"]
            )
            st.dataframe(history_df, height=300, use_container_width=True)
        else:
            st.write("No moves yet.")

        # Display game status
        st.subheader("Game Status")
        st.write(f"Current player: {st.session_state.board.current_player.capitalize()}")
        evaluation = st.session_state.board.evaluate_position_cached(
            st.session_state.board.get_board_hash())
        st.write(f"Position evaluation: {evaluation:.2f}")
        
        # Material count
        white_material = sum(PIECE_VALUES[piece.lower()] 
                           for row in st.session_state.board.board 
                           for piece in row if piece.isupper())
        black_material = sum(PIECE_VALUES[piece.lower()] 
                           for row in st.session_state.board.board 
                           for piece in row if piece.islower())
        st.write(f"Material balance: White {white_material} vs Black {black_material}")

        # Reset game button
        if st.button("Reset Game", key="reset_game"):
            st.session_state.board = ChessBoard()
            st.session_state.game_over = False
            st.session_state.selected_piece = None
            st.session_state.performance = {
                'positions_evaluated': 0,
                'cache_hits': 0,
                'avg_move_time': 0
            }
            st.experimental_rerun()

if __name__ == "__main__":
    main()