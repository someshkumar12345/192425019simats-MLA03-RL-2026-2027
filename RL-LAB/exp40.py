import numpy as np
import random
import matplotlib.pyplot as plt


# ============================================================
# PERSONALIZED EDUCATION ENVIRONMENT
# ============================================================

class EducationEnvironment:

    def __init__(self):

        # ----------------------------------------------------
        # Student knowledge levels
        # ----------------------------------------------------

        # 0 = Beginner
        # 1 = Intermediate
        # 2 = Advanced

        self.knowledge_levels = 3

        self.knowledge_names = {
            0: "Beginner",
            1: "Intermediate",
            2: "Advanced"
        }

        # ----------------------------------------------------
        # Curriculum difficulty
        # ----------------------------------------------------

        # 0 = Easy
        # 1 = Medium
        # 2 = Hard

        self.difficulty_levels = 3

        self.difficulty_names = {
            0: "Easy",
            1: "Medium",
            2: "Hard"
        }

        # ----------------------------------------------------
        # Student engagement
        # ----------------------------------------------------

        # 0 = Low
        # 1 = Medium
        # 2 = High

        self.engagement_levels = 3

        self.engagement_names = {
            0: "Low",
            1: "Medium",
            2: "High"
        }

        # ----------------------------------------------------
        # Teaching actions
        # ----------------------------------------------------

        # 0 = Easy Content
        # 1 = Moderate Content
        # 2 = Difficult Content
        # 3 = Revision
        # 4 = Practice Quiz

        self.actions = [
            "Easy Content",
            "Moderate Content",
            "Difficult Content",
            "Revision",
            "Practice Quiz"
        ]

        self.num_actions = len(
            self.actions
        )

        # Maximum learning sessions
        self.max_steps = 30

        self.reset()

    # ========================================================
    # RESET STUDENT
    # ========================================================

    def reset(self):

        # Random initial knowledge
        self.knowledge = random.choice(
            [0, 1, 2]
        )

        # Initial engagement
        self.engagement = random.choice(
            [0, 1, 2]
        )

        # Curriculum difficulty
        self.curriculum_difficulty = 1

        # Learning progress
        self.progress = 0

        # Number of completed lessons
        self.steps = 0

        # Student performance
        self.score = 0

        self.done = False

        return self.get_state()

    # ========================================================
    # STATE
    # ========================================================

    def get_state(self):

        return (
            self.knowledge,
            self.curriculum_difficulty,
            self.engagement
        )

    # ========================================================
    # STATE INDEX
    # ========================================================

    def state_to_index(
        self,
        state
    ):

        knowledge, difficulty, engagement = state

        return (
            knowledge * 9
            +
            difficulty * 3
            +
            engagement
        )

    # ========================================================
    # LEARNING PROBABILITY
    # ========================================================

    def learning_probability(
        self,
        action
    ):

        # ----------------------------------------------------
        # Difference between content difficulty
        # and student knowledge
        # ----------------------------------------------------

        if action == 0:

            content_difficulty = 0

        elif action == 1:

            content_difficulty = 1

        elif action == 2:

            content_difficulty = 2

        elif action == 3:

            # Revision is easier
            content_difficulty = max(
                0,
                self.knowledge - 1
            )

        else:

            # Quiz difficulty matches knowledge
            content_difficulty = self.knowledge

        difficulty_gap = abs(
            content_difficulty
            -
            self.knowledge
        )

        # ----------------------------------------------------
        # Base probability
        # ----------------------------------------------------

        probability = 0.80

        # Content too difficult
        if difficulty_gap == 1:

            probability -= 0.15

        elif difficulty_gap >= 2:

            probability -= 0.30

        # Low engagement reduces learning
        if self.engagement == 0:

            probability -= 0.15

        elif self.engagement == 2:

            probability += 0.10

        # Revision is generally easier
        if action == 3:

            probability += 0.10

        return np.clip(
            probability,
            0.05,
            0.95
        )

    # ========================================================
    # STEP
    # ========================================================

    def step(
        self,
        action
    ):

        self.steps += 1

        reward = 0

        probability = (
            self.learning_probability(
                action
            )
        )

        # ----------------------------------------------------
        # Learning outcome
        # ----------------------------------------------------

        learned = (
            random.random()
            <
            probability
        )

        # ====================================================
        # PRACTICE QUIZ
        # ====================================================

        if action == 4:

            # Quiz measures knowledge
            quiz_success = (
                random.random()
                <
                (
                    0.45
                    +
                    0.20
                    * self.knowledge
                )
            )

            if quiz_success:

                reward += 15

                self.score += 10

            else:

                reward -= 5

        # ====================================================
        # OTHER TEACHING ACTIONS
        # ====================================================

        else:

            if learned:

                # Learning improves knowledge
                if self.knowledge < 2:

                    self.knowledge += 1

                    self.progress += 1

                reward += 10

            else:

                reward -= 3

        # ====================================================
        # ENGAGEMENT UPDATE
        # ====================================================

        # Difficult material can reduce engagement
        if action == 2:

            if self.knowledge == 0:

                self.engagement = max(
                    0,
                    self.engagement - 1
                )

        # Revision can increase engagement
        elif action == 3:

            self.engagement = min(
                2,
                self.engagement + 1
            )

        # Successful learning increases engagement
        if learned:

            self.engagement = min(
                2,
                self.engagement + 1
            )

        # ----------------------------------------------------
        # Poor engagement penalty
        # ----------------------------------------------------

        if self.engagement == 0:

            reward -= 2

        # ----------------------------------------------------
        # Successful completion
        # ----------------------------------------------------

        if self.knowledge == 2:

            reward += 50

            self.done = True

        # ----------------------------------------------------
        # Time limit
        # ----------------------------------------------------

        if self.steps >= self.max_steps:

            self.done = True

        return (
            self.get_state(),
            reward,
            self.done,
            {
                "knowledge":
                    self.knowledge,

                "engagement":
                    self.engagement,

                "progress":
                    self.progress,

                "score":
                    self.score
            }
        )


# ============================================================
# Q-LEARNING PERSONALIZATION AGENT
# ============================================================

class QLearningAgent:

    def __init__(
        self,
        state_size,
        action_size
    ):

        self.state_size = state_size

        self.action_size = action_size

        # Learning rate
        self.alpha = 0.1

        # Discount factor
        self.gamma = 0.95

        # Exploration probability
        self.epsilon = 1.0

        self.epsilon_min = 0.05

        self.epsilon_decay = 0.995

        # Q-table
        self.q_table = np.zeros(
            (
                state_size,
                action_size
            )
        )

    # ========================================================
    # ACTION SELECTION
    # ========================================================

    def choose_action(
        self,
        state
    ):

        # Exploration
        if (
            random.random()
            <
            self.epsilon
        ):

            return random.randrange(
                self.action_size
            )

        # Exploitation
        return np.argmax(
            self.q_table[state]
        )

    # ========================================================
    # Q-LEARNING UPDATE
    # ========================================================

    def update(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        current_q = (
            self.q_table[
                state,
                action
            ]
        )

        if done:

            target = reward

        else:

            target = (
                reward
                +
                self.gamma
                *
                np.max(
                    self.q_table[
                        next_state
                    ]
                )
            )

        self.q_table[
            state,
            action
        ] = (
            current_q
            +
            self.alpha
            *
            (
                target
                -
                current_q
            )
        )

    # ========================================================
    # DECAY EXPLORATION
    # ========================================================

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon
            *
            self.epsilon_decay
        )


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

environment = EducationEnvironment()


# Number of possible states:
# Knowledge × Difficulty × Engagement

state_size = (
    environment.knowledge_levels
    *
    environment.difficulty_levels
    *
    environment.engagement_levels
)


agent = QLearningAgent(
    state_size=state_size,
    action_size=environment.num_actions
)


# ============================================================
# TRAINING
# ============================================================

episodes = 5000

reward_history = []

progress_history = []

score_history = []

success_history = []

engagement_history = []


print("=" * 75)
print(" PERSONALIZED EDUCATION USING REINFORCEMENT LEARNING")
print("=" * 75)

print("\nTraining started...\n")


for episode in range(
    episodes
):

    state_tuple = environment.reset()

    state = environment.state_to_index(
        state_tuple
    )

    total_reward = 0

    total_progress = 0

    total_score = 0

    success = 0

    engagement_sum = 0

    for step in range(
        environment.max_steps
    ):

        # ----------------------------------------------------
        # Select teaching intervention
        # ----------------------------------------------------

        action = agent.choose_action(
            state
        )

        # ----------------------------------------------------
        # Student/environment response
        # ----------------------------------------------------

        (
            next_state_tuple,
            reward,
            done,
            info
        ) = environment.step(
            action
        )

        next_state = (
            environment.state_to_index(
                next_state_tuple
            )
        )

        # ----------------------------------------------------
        # Q-learning update
        # ----------------------------------------------------

        agent.update(
            state,
            action,
            reward,
            next_state,
            done
        )

        state = next_state

        total_reward += reward

        total_progress += (
            info["progress"]
        )

        total_score += (
            info["score"]
        )

        engagement_sum += (
            info["engagement"]
        )

        if (
            info["knowledge"]
            == 2
        ):

            success = 1

        if done:

            break

    # --------------------------------------------------------
    # Reduce exploration
    # --------------------------------------------------------

    agent.decay_epsilon()

    reward_history.append(
        total_reward
    )

    progress_history.append(
        total_progress
    )

    score_history.append(
        total_score
    )

    success_history.append(
        success
    )

    engagement_history.append(
        engagement_sum /
        (step + 1)
    )

    # --------------------------------------------------------
    # Display progress
    # --------------------------------------------------------

    if (
        (episode + 1)
        % 500
        == 0
    ):

        avg_reward = np.mean(
            reward_history[-500:]
        )

        success_rate = (
            np.mean(
                success_history[-500:]
            )
            *
            100
        )

        avg_engagement = np.mean(
            engagement_history[-500:]
        )

        print(
            f"Episode {episode + 1:4d} | "
            f"Avg Reward: "
            f"{avg_reward:8.2f} | "
            f"Success Rate: "
            f"{success_rate:6.2f}% | "
            f"Avg Engagement: "
            f"{avg_engagement:.2f} | "
            f"Epsilon: "
            f"{agent.epsilon:.3f}"
        )


# ============================================================
# TRAINING RESULTS
# ============================================================

print("\n" + "=" * 75)
print(" TRAINING RESULTS")
print("=" * 75)

print(
    "Training Episodes:",
    episodes
)

print(
    "Number of States:",
    state_size
)

print(
    "Number of Actions:",
    environment.num_actions
)

print(
    "Final Average Reward:",
    round(
        np.mean(
            reward_history[-500:]
        ),
        2
    )
)

print(
    "Final Success Rate:",
    round(
        np.mean(
            success_history[-500:]
        )
        *
        100,
        2
    ),
    "%"
)

print(
    "Final Epsilon:",
    round(
        agent.epsilon,
        4
    )
)


# ============================================================
# EVALUATION
# ============================================================

print("\n" + "=" * 75)
print(" EVALUATING PERSONALIZED POLICY")
print("=" * 75)

# Disable exploration
agent.epsilon = 0

evaluation_episodes = 500

evaluation_rewards = []

evaluation_progress = []

evaluation_scores = []

evaluation_engagement = []

evaluation_success = 0


for episode in range(
    evaluation_episodes
):

    state_tuple = environment.reset()

    state = environment.state_to_index(
        state_tuple
    )

    total_reward = 0

    total_progress = 0

    total_score = 0

    engagement_sum = 0

    for step in range(
        environment.max_steps
    ):

        # Choose best learned action
        action = agent.choose_action(
            state
        )

        (
            next_state_tuple,
            reward,
            done,
            info
        ) = environment.step(
            action
        )

        state = environment.state_to_index(
            next_state_tuple
        )

        total_reward += reward

        total_progress += (
            info["progress"]
        )

        total_score += (
            info["score"]
        )

        engagement_sum += (
            info["engagement"]
        )

        if done:

            break

    evaluation_rewards.append(
        total_reward
    )

    evaluation_progress.append(
        total_progress
    )

    evaluation_scores.append(
        total_score
    )

    evaluation_engagement.append(
        engagement_sum /
        (step + 1)
    )

    if info["knowledge"] == 2:

        evaluation_success += 1


# ============================================================
# EVALUATION RESULTS
# ============================================================

success_rate = (
    evaluation_success
    /
    evaluation_episodes
    *
    100
)


print(
    "Evaluation Episodes:",
    evaluation_episodes
)

print(
    "Students Reaching Advanced Level:",
    evaluation_success
)

print(
    "Learning Success Rate:",
    round(
        success_rate,
        2
    ),
    "%"
)

print(
    "Average Reward:",
    round(
        np.mean(
            evaluation_rewards
        ),
        2
    )
)

print(
    "Average Learning Progress:",
    round(
        np.mean(
            evaluation_progress
        ),
        2
    )
)

print(
    "Average Student Score:",
    round(
        np.mean(
            evaluation_scores
        ),
        2
    )
)

print(
    "Average Engagement:",
    round(
        np.mean(
            evaluation_engagement
        ),
        2
    )
)


# ============================================================
# LEARNED PERSONALIZED POLICY
# ============================================================

print("\n" + "=" * 75)
print(" LEARNED PERSONALIZED TEACHING POLICY")
print("=" * 75)


for knowledge in range(3):

    for difficulty in range(3):

        for engagement in range(3):

            state_tuple = (
                knowledge,
                difficulty,
                engagement
            )

            state = (
                environment.state_to_index(
                    state_tuple
                )
            )

            best_action = np.argmax(
                agent.q_table[state]
            )

            print(
                f"Knowledge: "
                f"{environment.knowledge_names[knowledge]:<12} | "
                f"Curriculum: "
                f"{environment.difficulty_names[difficulty]:<12} | "
                f"Engagement: "
                f"{environment.engagement_names[engagement]:<12} | "
                f"Recommended: "
                f"{environment.actions[best_action]}"
            )


# ============================================================
# SAMPLE STUDENT SIMULATION
# ============================================================

print("\n" + "=" * 75)
print(" SAMPLE PERSONALIZED LEARNING SESSION")
print("=" * 75)


state_tuple = environment.reset()

state = environment.state_to_index(
    state_tuple
)

print(
    "\nInitial Student State:"
)

print(
    "Knowledge:",
    environment.knowledge_names[
        environment.knowledge
    ]
)

print(
    "Engagement:",
    environment.engagement_names[
        environment.engagement
    ]
)


for step in range(
    environment.max_steps
):

    action = agent.choose_action(
        state
    )

    print(
        f"\nStep {step + 1}"
    )

    print(
        "Knowledge:",
        environment.knowledge_names[
            environment.knowledge
        ]
    )

    print(
        "Engagement:",
        environment.engagement_names[
            environment.engagement
        ]
    )

    print(
        "Recommended Content:",
        environment.actions[action]
    )

    (
        next_state_tuple,
        reward,
        done,
        info
    ) = environment.step(
        action
    )

    print(
        "Reward:",
        reward
    )

    print(
        "Progress:",
        info["progress"]
    )

    state = environment.state_to_index(
        next_state_tuple
    )

    if done:

        break


print(
    "\nFinal Knowledge Level:",
    environment.knowledge_names[
        environment.knowledge
    ]
)

print(
    "Final Score:",
    environment.score
)


# ============================================================
# REWARD GRAPH
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    reward_history
)

plt.xlabel(
    "Episode"
)

plt.ylabel(
    "Total Reward"
)

plt.title(
    "Personalized Education RL Training"
)

plt.grid(True)

plt.show()


# ============================================================
# SUCCESS RATE GRAPH
# ============================================================

window = 100

success_curve = (
    np.convolve(
        success_history,
        np.ones(window) / window,
        mode="valid"
    )
    *
    100
)


plt.figure(
    figsize=(10, 5)
)

plt.plot(
    success_curve
)

plt.xlabel(
    "Episode"
)

plt.ylabel(
    "Learning Success Rate (%)"
)

plt.title(
    "Student Learning Success During Training"
)

plt.grid(True)

plt.show()


# ============================================================
# ENGAGEMENT GRAPH
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    engagement_history
)

plt.xlabel(
    "Episode"
)

plt.ylabel(
    "Average Engagement Level"
)

plt.title(
    "Student Engagement During Training"
)

plt.grid(True)

plt.show()
