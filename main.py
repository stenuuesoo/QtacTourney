import random
import sqlite3
import time
import threading

class RandomAgent:
    def move(self, board):
        empty_cells = [(i, j) for i in range(3) for j in range(3) if board[i][j] == " "]
        return random.choice(empty_cells)


class QAgent:
    def __init__(self, conn, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.995):
        self.conn = conn
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay

    def move(self, board):
        board_str = str(board)
        c = self.conn.cursor()

        # Fetch Q-values
        c.execute("SELECT action, q_value FROM q_values WHERE state=?", (board_str,))
        entries = c.fetchall()

        # If state is new, initialize all possible actions with Q-value 0
        if not entries:
            for i in range(3):
                for j in range(3):
                    if board[i][j] == " ":
                        c.execute("INSERT INTO q_values (state, action, q_value) VALUES (?, ?, ?)",
                                  (board_str, str((i, j)), 0))
            self.conn.commit()
            # Fetch Q-values again
            c.execute("SELECT action, q_value FROM q_values WHERE state=?", (board_str,))
            entries = c.fetchall()

        # Epsilon-greedy policy for move selection
        if random.uniform(0, 1) < self.exploration_rate:
            move = random.choice([(i, j) for i in range(3) for j in range(3) if board[i][j] == " "])
        else:
            move = max(entries, key=lambda x: x[1])[0]
            move = eval(move)  # Convert string representation back to tuple

        # Decay exploration rate
        self.exploration_rate *= self.exploration_decay

        return move

    def learn(self, old_state, action_taken, reward, new_state):
        c = self.conn.cursor()

        # Get old Q-value
        c.execute("SELECT q_value FROM q_values WHERE state=? AND action=?", (old_state, str(action_taken)))
        old_q_value_entry = c.fetchone()

        old_q_value = old_q_value_entry[0] if old_q_value_entry else 0

        # Get max Q-value for new state
        c.execute("SELECT MAX(q_value) FROM q_values WHERE state=?", (new_state,))
        max_new_q_value_entry = c.fetchone()

        max_new_q_value = max_new_q_value_entry[0] if max_new_q_value_entry and max_new_q_value_entry[
            0] is not None else 0

        # Calculate new Q-value
        new_q_value = old_q_value + self.learning_rate * (reward + self.discount_factor * max_new_q_value - old_q_value)

        # Update the Q-value in the database
        if old_q_value_entry:
            c.execute("UPDATE q_values SET q_value=? WHERE state=? AND action=?",
                      (new_q_value, old_state, str(action_taken)))
        else:
            c.execute("INSERT INTO q_values (state, action, q_value) VALUES (?, ?, ?)",
                      (old_state, str(action_taken), new_q_value))
        self.conn.commit()


class Statistics:
    def __init__(self):
        self.agent_results = {
            "QAgent": 0,
            "MinMaxAgent": 0,
            "RandomAgent": 0,
            "Draws": 0
        }
        self.game_outcomes = []
        self.board_frequencies = {}
        self.draw_boards = []

    def record_outcome(self, outcome, board):
        self.game_outcomes.append(outcome)
        if outcome == 'D':
            self.draw_boards.append(board)

    def record_board(self, board):
        board_str = str(board)
        if board_str not in self.board_frequencies:
            self.board_frequencies[board_str] = 0
        self.board_frequencies[board_str] += 1

    def win_percentage(self):
        x_wins = self.game_outcomes.count('X')
        total_games = len(self.game_outcomes)
        return x_wins / total_games * 100 if total_games else 0

    def top_boards(self, n=3):
        sorted_boards = sorted(self.board_frequencies.items(), key=lambda x: x[1], reverse=True)
        return sorted_boards[:n]

    def display_board(self, board):
        rows = []
        for row in board:
            rows.append("|".join(row))
        return "\n".join(rows)

    def print_statistics(self):
        print(f"Win Percentage for X over time: {self.win_percentage()}%")

        top_draws = self.top_boards_for_outcome('D', 2)
        top_wins = self.top_boards_for_outcome('X', 2)

        print("\nTop 2 draw boards vs. Top 2 win boards:")
        for (draw_board, draw_freq), (win_board, win_freq) in zip(top_draws, top_wins):
            draw_board_disp = self.display_board(eval(draw_board))
            win_board_disp = self.display_board(eval(win_board))

            combined_board = "\n".join([f"{d_row:<10}{w_row}" for d_row, w_row in
                                        zip(draw_board_disp.split('\n'), win_board_disp.split('\n'))])

            print(combined_board)
            print(f"Draws: {draw_freq} times   Wins: {win_freq} times")
            print()

    def top_boards_for_outcome(self, outcome, n=3):
        relevant_boards = [board for board, _ in zip(self.board_frequencies.keys(), self.game_outcomes) if _ == outcome]
        sorted_boards = sorted(((board, self.board_frequencies[board]) for board in relevant_boards),
                               key=lambda x: x[1], reverse=True)
        return sorted_boards[:n]

    def record_tournament_outcome(self, winner_class_name):
        if winner_class_name:
            self.agent_results[winner_class_name] += 1
        else:
            self.agent_results["Draws"] += 1

    def print_tournament_statistics(self):
        for agent, count in self.agent_results.items():
            print(f"{agent}: {count} games won")


class MinMaxAgent:
    def move(self, board):
        move, _ = self.minmax(board, 'X', 0)
        return move

    def minmax(self, board, player, depth):
        if check_win(board):
            if player == 'O':  # Last move was X's
                return (None, 10 - depth)  # Subtract depth for quicker wins
            else:  # Last move was O's
                return (None, -10 + depth)  # Add depth to delay loss

        if is_full(board):
            return (None, 0)

        moves = [(i, j) for i in range(3) for j in range(3) if board[i][j] == " "]
        best_move = None

        if player == 'X':
            best_score = -float('inf')
            for move in moves:
                board[move[0]][move[1]] = 'X'
                _, score = self.minmax(board, 'O', depth + 1)
                board[move[0]][move[1]] = ' '
                if score > best_score:
                    best_score = score
                    best_move = move
        else:
            best_score = float('inf')
            for move in moves:
                board[move[0]][move[1]] = 'O'
                _, score = self.minmax(board, 'X', depth + 1)
                board[move[0]][move[1]] = ' '
                if score < best_score:
                    best_score = score
                    best_move = move

        return best_move, best_score


# Note: The MinMaxAgent class considers that X always wants to maximize the score and O always wants to minimize.


class DetailedStatistics(Statistics):
    def __init__(self):
        super().__init__()
        self.game_details = []  # Captures move sequences and game results

    def record_game_details(self, winner, moves, final_board):
        self.game_details.append({
            'winner': winner.__class__.__name__ if winner else 'Draw',
            'moves': moves,
            'final_board': final_board
        })

    def print_detailed_statistics(self):
        for game_number, game in enumerate(self.game_details, start=1):
            print(f"\nGame Number: {game_number}")
            print(f"Winner: {game['winner']}")
            print("Move Sequence:")
            for idx, move in enumerate(game['moves'], start=1):
                print(f"{idx}. {move[0].__class__.__name__} moved to {move[1]}")
                print("Board State:")
                print(self.display_board(move[2]))
            print("------------------------")

    def print_tournament_outcome(self):
        game_counts = {agent: 0 for agent in self.agent_results}
        for game in self.game_details:
            if game['winner'] != 'Draw':
                game_counts[game['winner']] += 1
        print("\nTournament Outcome:")
        for agent, count in game_counts.items():
            print(f"{agent}: {count} games won")

    def print_additional_statistics(self):
        self.print_detailed_statistics()  # Print the detailed statistics as before

        # Additional Statistics
        print("\nAdditional Statistics:")
        print(f"Number of Draws: {self.game_outcomes.count('D')}")
        print(
            f"Average Game Length: {sum(len(game['moves']) for game in self.game_details) / len(self.game_details):.2f} moves")
        print(f"Top 3 Draw Boards: {', '.join(self.top_boards_for_outcome('D', 3))}")
        print(f"Top 3 Win Boards: {', '.join(self.top_boards_for_outcome('X', 3))}")


def init_database():
    conn = sqlite3.connect(':memory:')
    c = conn.cursor()

    c.execute('''CREATE TABLE q_values
                 (state TEXT, action TEXT, q_value REAL)''')
    conn.commit()

    return conn

conn = init_database()



def check_win(board):
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != " ":
            return True
        if board[0][i] == board[1][i] == board[2][i] != " ":
            return True
    if board[0][0] == board[1][1] == board[2][2] != " ":
        return True
    if board[0][2] == board[1][1] == board[2][0] != " ":
        return True
    return False


def is_full(board):
    for row in board:
        for cell in row:
            if cell == " ":
                return False
    return True


def play_game(agent1, agent2, start='X'):
    board = [[" "] * 3 for _ in range(3)]
    current_agent = agent1 if start == 'X' else agent2
    other_agent = agent2 if start == 'X' else agent1
    last_move = None
    move_sequence = []

    while not check_win(board) and not is_full(board):
        move = current_agent.move(board)
        new_board = [row.copy() for row in board]  # deep copy of the board state
        new_board[move[0]][move[1]] = "X" if current_agent == agent1 else "O"
        move_sequence.append((current_agent, move, new_board))
        last_move = move
        board = new_board
        if check_win(board):
            return current_agent, board, last_move, move_sequence
        current_agent, other_agent = other_agent, current_agent

    return None, board, last_move, move_sequence


def display_results(tournament_stats):
    while True:
        time.sleep(1)  # refresh every second
        tournament_stats.print_detailed_statistics()


def play_tournament():
    agents = [QAgent(conn, exploration_rate=0), MinMaxAgent(), RandomAgent()]
    tournament_stats = DetailedStatistics()

    total_runtime = 0  # for calculating total runtime

    # start a thread to display results
    thread = threading.Thread(target=display_results, args=(tournament_stats,))
    thread.daemon = True  # this makes sure the thread exits when main program exits
    thread.start()

    for game in range(1000):
        start = 'X' if game % 2 == 0 else 'O'
        random.shuffle(agents)  # To randomize starting order

        start_time = time.time()
        winner, final_board, _, moves = play_game(agents[0], agents[1], start=start)
        end_time = time.time()

        game_runtime = end_time - start_time
        total_runtime += game_runtime

        tournament_stats.record_game_details(winner, moves, final_board)
        if winner:
            tournament_stats.record_tournament_outcome(winner.__class__.__name__)
        else:
            tournament_stats.record_tournament_outcome(None)

        # Add a sleep if you want to see results more clearly
        time.sleep(0.1)

    avg_runtime = total_runtime / 1000
    print(f"Total runtime: {total_runtime:.2f} seconds")
    print(f"Average runtime per game: {avg_runtime:.2f} seconds")
    print(f"Total games run: 1000")  # Add this line

    tournament_stats.print_tournament_statistics()


play_tournament()


