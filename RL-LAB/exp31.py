import numpy as np
import random

# ============================================================
# SARSA AGENT FOR TIC-TAC-TOE
# ============================================================

class TicTacToe:

    def __init__(self):
        self.reset()

    def reset(self):
        # 0 = Empty
        # 1 = AI
        # -1 = Opponent
        self.board = [0] * 9
        return self.get_state()

    def get_state(self):
        return tuple(self.board)

    def available_actions(self):
        return [
            i for i in range(9)
            if self.board[i] == 0
        ]

    def make_move(self, action, player):
        self.board[action] = player

    def check_winner(self):

        winning_combinations = [
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),
            (0, 4, 8),
            (2, 4, 6)
        ]

        for a, b, c in winning_combinations:

            if (
                self.board[a] != 0
                and
                self.board[a] ==
                self.board[b]
                and
                self.board[b] ==
                self.board[c]
            ):
                return self.board[a]

        if 0 not in self.board:
            return 0  # Draw

        return None  # Game continues


# ============================================================
# SARSA AGENT
# ============================================================

class SARSAAgent:

    def __init__(
        self,
        alpha=0.1,
        gamma=0.9,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.9995
    ):

        # Q-table:
        # Dictionary:
        # state -> action -> Q-value
        self.q_table = {}

        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    # --------------------------------------------------------
    # Get Q-values for a state
    # --------------------------------------------------------

    def get_q_values(self, state):

        if state not in self.q_table:

            self.q_table[state] = np.zeros(9)

        return self.q_table[state]

    # --------------------------------------------------------
    # Choose action using epsilon-greedy policy
    # --------------------------------------------------------

    def choose_action(
        self,
        state,
        available_actions,
        training=True
    ):

        q_values = self.get_q_values(state)

        # Exploration
        if (
            training
            and
            random.random() < self.epsilon
        ):

            return random.choice(
                available_actions
            )

        # Exploitation
        available_q_values = [
            q_values[action]
            for action in available_actions
        ]

        max_q = max(
            available_q_values
        )

        best_actions = [
            action
            for action in available_actions
            if q_values[action] == max_q
        ]

        return random.choice(
            best_actions
        )

    # --------------------------------------------------------
    # SARSA update
    # --------------------------------------------------------

    def update(
        self,
        state,
        action,
        reward,
        next_state,
        next_action,
        done
    ):

        q_values = self.get_q_values(state)

        current_q = q_values[action]

        if done:

            target = reward

        else:

            next_q_values = self.get_q_values(
                next_state
            )

            # SARSA uses the ACTUAL next action
            target = (
                reward
                +
                self.gamma
                * next_q_values[next_action]
            )

        # SARSA update equation
        q_values[action] = (
            current_q
            +
            self.alpha
            * (target - current_q)
        )

    # --------------------------------------------------------
    # Decay exploration
    # --------------------------------------------------------

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )


# ============================================================
# RANDOM OPPONENT
# ============================================================

def opponent_move(game):

    available = game.available_actions()

    if available:

        return random.choice(
            available
        )

    return None


# ============================================================
# TRAINING
# ============================================================

agent = SARSAAgent()

episodes = 50000

wins = 0
losses = 0
draws = 0

training_rewards = []

print("=" * 60)
print(" SARSA TIC-TAC-TOE")
print("=" * 60)

print("\nTraining started...\n")


for episode in range(episodes):

    game = TicTacToe()

    state = game.reset()

    total_reward = 0

    # --------------------------------------------------------
    # AI selects first action
    # --------------------------------------------------------

    available = game.available_actions()

    action = agent.choose_action(
        state,
        available,
        training=True
    )

    while True:

        # ----------------------------------------------------
        # AI move
        # ----------------------------------------------------

        game.make_move(
            action,
            1
        )

        result = game.check_winner()

        # AI wins
        if result == 1:

            reward = 1

            wins += 1

            next_state = game.get_state()

            agent.update(
                state,
                action,
                reward,
                next_state,
                None,
                True
            )

            total_reward += reward

            break

        # Draw
        if result == 0:

            reward = 0

            draws += 1

            next_state = game.get_state()

            agent.update(
                state,
                action,
                reward,
                next_state,
                None,
                True
            )

            break

        # ----------------------------------------------------
        # Opponent move
        # ----------------------------------------------------

        opponent_action = opponent_move(game)

        if opponent_action is None:

            reward = 0

            draws += 1

            break

        game.make_move(
            opponent_action,
            -1
        )

        result = game.check_winner()

        # Opponent wins
        if result == -1:

            reward = -1

            losses += 1

            next_state = game.get_state()

            agent.update(
                state,
                action,
                reward,
                next_state,
                None,
                True
            )

            total_reward += reward

            break

        # Draw after opponent move
        if result == 0:

            reward = 0

            draws += 1

            next_state = game.get_state()

            agent.update(
                state,
                action,
                reward,
                next_state,
                None,
                True
            )

            break

        # ----------------------------------------------------
        # Next state and next action
        # ----------------------------------------------------

        next_state = game.get_state()

        available = game.available_actions()

        next_action = agent.choose_action(
            next_state,
            available,
            training=True
        )

        # ----------------------------------------------------
        # SARSA UPDATE
        # ----------------------------------------------------

        reward = 0

        agent.update(
            state,
            action,
            reward,
            next_state,
            next_action,
            False
        )

        state = next_state

        action = next_action

        total_reward += reward

    # Reduce exploration
    agent.decay_epsilon()

    training_rewards.append(
        total_reward
    )

    # Display training progress
    if (episode + 1) % 5000 == 0:

        print(
            f"Episode {episode + 1:5d} | "
            f"Epsilon: {agent.epsilon:.4f} | "
            f"States Learned: {len(agent.q_table)}"
        )


# ============================================================
# TRAINING RESULTS
# ============================================================

print("\n" + "=" * 60)
print(" TRAINING RESULTS")
print("=" * 60)

print(
    "Episodes:",
    episodes
)

print(
    "Wins:",
    wins
)

print(
    "Losses:",
    losses
)

print(
    "Draws:",
    draws
)

print(
    "States Learned:",
    len(agent.q_table)
)

print(
    "Final Epsilon:",
    round(
        agent.epsilon,
        4
    )
)


# ============================================================
# DISPLAY BOARD
# ============================================================

def print_board(board):

    symbols = {
        0: " ",
        1: "X",
        -1: "O"
    }

    print()

    for row in range(3):

        cells = []

        for col in range(3):

            index = row * 3 + col

            cells.append(
                symbols[board[index]]
            )

        print(
            f" {cells[0]} | {cells[1]} | {cells[2]} "
        )

        if row < 2:
            print("---+---+---")

    print()


# ============================================================
# EVALUATION
# ============================================================

print("\n" + "=" * 60)
print(" EVALUATION")
print("=" * 60)

# Disable exploration
agent.epsilon = 0

evaluation_games = 1000

eval_wins = 0
eval_losses = 0
eval_draws = 0


for game_number in range(
    evaluation_games
):

    game = TicTacToe()

    state = game.reset()

    while True:

        # ----------------------------------------------------
        # AI move
        # ----------------------------------------------------

        available = game.available_actions()

        action = agent.choose_action(
            state,
            available,
            training=False
        )

        game.make_move(
            action,
            1
        )

        result = game.check_winner()

        if result == 1:

            eval_wins += 1

            break

        if result == 0:

            eval_draws += 1

            break

        # ----------------------------------------------------
        # Random opponent move
        # ----------------------------------------------------

        opponent_action = opponent_move(
            game
        )

        if opponent_action is None:

            eval_draws += 1

            break

        game.make_move(
            opponent_action,
            -1
        )

        result = game.check_winner()

        if result == -1:

            eval_losses += 1

            break

        if result == 0:

            eval_draws += 1

            break

        state = game.get_state()


# ============================================================
# EVALUATION RESULTS
# ============================================================

print(
    "Evaluation Games:",
    evaluation_games
)

print(
    "Wins:",
    eval_wins
)

print(
    "Losses:",
    eval_losses
)

print(
    "Draws:",
    eval_draws
)

print(
    "Win Rate:",
    round(
        eval_wins /
        evaluation_games *
        100,
        2
    ),
    "%"
)

print(
    "Loss Rate:",
    round(
        eval_losses /
        evaluation_games *
        100,
        2
    ),
    "%"
)

print(
    "Draw Rate:",
    round(
        eval_draws /
        evaluation_games *
        100,
        2
    ),
    "%"
)


# ============================================================
# DEMONSTRATE ONE GAME
# ============================================================

print("\n" + "=" * 60)
print(" SAMPLE GAME")
print("=" * 60)

game = TicTacToe()

state = game.reset()

print("\nInitial Board:")

print_board(
    game.board
)

while True:

    available = game.available_actions()

    action = agent.choose_action(
        state,
        available,
        training=False
    )

    game.make_move(
        action,
        1
    )

    print(
        "AI chooses position:",
        action + 1
    )

    print_board(
        game.board
    )

    result = game.check_winner()

    if result == 1:

        print("AI WINS!")

        break

    if result == 0:

        print("DRAW!")

        break

    # Opponent
    opponent_action = opponent_move(
        game
    )

    game.make_move(
        opponent_action,
        -1
    )

    print(
        "Opponent chooses position:",
        opponent_action + 1
    )

    print_board(
        game.board
    )

    result = game.check_winner()

    if result == -1:

        print("OPPONENT WINS!")

        break

    if result == 0:

        print("DRAW!")

        break

    state = game.get_state()
