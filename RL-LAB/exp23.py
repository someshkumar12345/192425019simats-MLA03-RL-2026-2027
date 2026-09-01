import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt


# ============================================================
# HIGHWAY ENVIRONMENT
# ============================================================

class HighwayEnv:

    def __init__(self):
        self.num_lanes = 3
        self.max_position = 100
        self.max_speed = 10

        # Actions:
        # 0 = Keep Lane
        # 1 = Change Left
        # 2 = Change Right
        # 3 = Accelerate
        # 4 = Brake

        self.num_actions = 5

        self.reset()

    def reset(self):

        self.position = 0.0
        self.speed = 5.0
        self.lane = 1

        # Other vehicles
        self.vehicles = [
            {"lane": 1, "position": 25, "speed": 3},
            {"lane": 0, "position": 45, "speed": 7},
            {"lane": 2, "position": 35, "speed": 8},
            {"lane": 1, "position": 60, "speed": 4}
        ]

        self.done = False

        return self.get_state()

    # --------------------------------------------------------
    # State representation
    # --------------------------------------------------------

    def get_state(self):

        # Distance to nearest vehicle in each lane
        lane_distances = []

        for lane in range(self.num_lanes):

            distances = [
                v["position"] - self.position
                for v in self.vehicles
                if v["lane"] == lane
                and v["position"] > self.position
            ]

            if len(distances) == 0:
                distance = 50
            else:
                distance = min(distances)

            distance = min(distance, 50)

            lane_distances.append(distance / 50.0)

        state = np.array([
            self.position / self.max_position,
            self.speed / self.max_speed,
            self.lane / (self.num_lanes - 1),
            lane_distances[0],
            lane_distances[1],
            lane_distances[2]
        ], dtype=np.float32)

        return state

    # --------------------------------------------------------
    # Environment step
    # --------------------------------------------------------

    def step(self, action):

        reward = 0
        collision = False

        old_position = self.position

        # ----------------------------------------------------
        # Perform action
        # ----------------------------------------------------

        if action == 0:
            # Keep lane
            pass

        elif action == 1:
            # Change left
            if self.lane > 0:
                self.lane -= 1
            else:
                reward -= 5

        elif action == 2:
            # Change right
            if self.lane < self.num_lanes - 1:
                self.lane += 1
            else:
                reward -= 5

        elif action == 3:
            # Accelerate
            self.speed += 1

        elif action == 4:
            # Brake
            self.speed -= 1

        self.speed = np.clip(self.speed, 0, self.max_speed)

        # ----------------------------------------------------
        # Move vehicle
        # ----------------------------------------------------

        self.position += self.speed

        # ----------------------------------------------------
        # Move traffic vehicles
        # ----------------------------------------------------

        for vehicle in self.vehicles:
            vehicle["position"] += vehicle["speed"]

        # ----------------------------------------------------
        # Collision detection
        # ----------------------------------------------------

        for vehicle in self.vehicles:

            distance = abs(
                vehicle["position"] - self.position
            )

            if (
                vehicle["lane"] == self.lane
                and distance < 3
            ):
                collision = True
                reward -= 100
                self.done = True

        # ----------------------------------------------------
        # Reward for forward progress
        # ----------------------------------------------------

        progress = self.position - old_position

        reward += progress * 0.5

        # ----------------------------------------------------
        # Reward for reaching destination
        # ----------------------------------------------------

        if self.position >= self.max_position:

            reward += 100
            self.done = True

        # ----------------------------------------------------
        # Penalty for unnecessary lane changes
        # ----------------------------------------------------

        if action in [1, 2]:
            reward -= 0.5

        # ----------------------------------------------------
        # Encourage maintaining reasonable speed
        # ----------------------------------------------------

        if self.speed < 2:
            reward -= 2

        return (
            self.get_state(),
            reward,
            self.done,
            {
                "collision": collision,
                "position": self.position,
                "lane": self.lane,
                "speed": self.speed
            }
        )


# ============================================================
# PPO ACTOR-CRITIC NETWORK
# ============================================================

class ActorCritic(nn.Module):

    def __init__(self, state_size, action_size):

        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Actor: action probabilities
        self.actor = nn.Linear(64, action_size)

        # Critic: state value
        self.critic = nn.Linear(64, 1)

    def forward(self, state):

        features = self.shared(state)

        action_logits = self.actor(features)

        value = self.critic(features)

        return action_logits, value


# ============================================================
# PPO AGENT
# ============================================================

class PPOAgent:

    def __init__(self, state_size, action_size):

        self.model = ActorCritic(
            state_size,
            action_size
        )

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.0003
        )

        self.gamma = 0.99
        self.lam = 0.95
        self.clip_epsilon = 0.2

        self.epochs = 10

    # --------------------------------------------------------
    # Select action
    # --------------------------------------------------------

    def select_action(self, state):

        state_tensor = torch.FloatTensor(
            state
        ).unsqueeze(0)

        logits, value = self.model(
            state_tensor
        )

        distribution = torch.distributions.Categorical(
            logits=logits
        )

        action = distribution.sample()

        log_probability = distribution.log_prob(
            action
        )

        return (
            action.item(),
            log_probability.item(),
            value.item()
        )

    # --------------------------------------------------------
    # Calculate advantages
    # --------------------------------------------------------

    def calculate_advantages(
        self,
        rewards,
        values,
        dones
    ):

        advantages = []
        advantage = 0

        values = values + [0]

        for t in reversed(range(len(rewards))):

            delta = (
                rewards[t]
                + self.gamma * values[t + 1] * (1 - dones[t])
                - values[t]
            )

            advantage = (
                delta
                + self.gamma
                * self.lam
                * (1 - dones[t])
                * advantage
            )

            advantages.insert(
                0,
                advantage
            )

        return np.array(advantages, dtype=np.float32)

    # --------------------------------------------------------
    # PPO update
    # --------------------------------------------------------

    def update(
        self,
        states,
        actions,
        old_log_probs,
        rewards,
        values,
        dones
    ):

        advantages = self.calculate_advantages(
            rewards,
            values,
            dones
        )

        returns = advantages + np.array(
            values,
            dtype=np.float32
        )

        states = torch.FloatTensor(
            np.array(states)
        )

        actions = torch.LongTensor(
            actions
        )

        old_log_probs = torch.FloatTensor(
            old_log_probs
        )

        advantages = torch.FloatTensor(
            advantages
        )

        returns = torch.FloatTensor(
            returns
        )

        # Normalize advantages
        advantages = (
            advantages - advantages.mean()
        ) / (advantages.std() + 1e-8)

        # ----------------------------------------------------
        # PPO optimization
        # ----------------------------------------------------

        for _ in range(self.epochs):

            logits, predicted_values = self.model(
                states
            )

            distribution = torch.distributions.Categorical(
                logits=logits
            )

            new_log_probs = distribution.log_prob(
                actions
            )

            entropy = distribution.entropy().mean()

            # Probability ratio
            ratio = torch.exp(
                new_log_probs - old_log_probs
            )

            # PPO clipped objective
            objective1 = (
                ratio * advantages
            )

            objective2 = (
                torch.clamp(
                    ratio,
                    1 - self.clip_epsilon,
                    1 + self.clip_epsilon
                )
                * advantages
            )

            actor_loss = -torch.min(
                objective1,
                objective2
            ).mean()

            # Critic loss
            critic_loss = nn.MSELoss()(
                predicted_values.squeeze(),
                returns
            )

            # Total loss
            loss = (
                actor_loss
                + 0.5 * critic_loss
                - 0.01 * entropy
            )

            self.optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                0.5
            )

            self.optimizer.step()


# ============================================================
# TRAINING
# ============================================================

env = HighwayEnv()

agent = PPOAgent(
    state_size=6,
    action_size=5
)

episodes = 500

reward_history = []
success_count = 0
collision_count = 0


print("============================================")
print(" PPO AUTONOMOUS HIGHWAY LANE CHANGING")
print("============================================")

print("\nTraining started...\n")


for episode in range(episodes):

    state = env.reset()

    states = []
    actions = []
    log_probs = []
    rewards = []
    values = []
    dones = []

    total_reward = 0

    for step in range(50):

        action, log_prob, value = agent.select_action(
            state
        )

        next_state, reward, done, info = env.step(
            action
        )

        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        values.append(value)
        dones.append(float(done))

        state = next_state

        total_reward += reward

        if done:

            if info["collision"]:
                collision_count += 1
            else:
                success_count += 1

            break

    # PPO update
    agent.update(
        states,
        actions,
        log_probs,
        rewards,
        values,
        dones
    )

    reward_history.append(
        total_reward
    )

    # Display progress
    if (episode + 1) % 50 == 0:

        avg_reward = np.mean(
            reward_history[-50:]
        )

        print(
            f"Episode {episode + 1:4d} | "
            f"Average Reward: {avg_reward:8.2f}"
        )


# ============================================================
# TRAINING RESULTS
# ============================================================

print("\n============================================")
print(" TRAINING RESULTS")
print("============================================")

print(
    "Episodes          :",
    episodes
)

print(
    "Successful Trips  :",
    success_count
)

print(
    "Collisions        :",
    collision_count
)

print(
    "Final Avg Reward  :",
    round(
        np.mean(reward_history[-50:]),
        2
    )
)


# ============================================================
# EVALUATION
# ============================================================

print("\n============================================")
print(" EVALUATION")
print("============================================")

# During evaluation we choose the best action
# instead of sampling randomly.

def evaluate_agent(env, agent):

    state = env.reset()

    path = []

    total_reward = 0

    for step in range(50):

        state_tensor = torch.FloatTensor(
            state
        ).unsqueeze(0)

        with torch.no_grad():

            logits, value = agent.model(
                state_tensor
            )

            action = torch.argmax(
                logits,
                dim=1
            ).item()

        next_state, reward, done, info = env.step(
            action
        )

        path.append({
            "step": step + 1,
            "position": round(
                info["position"], 2
            ),
            "lane": info["lane"],
            "speed": info["speed"],
            "action": action,
            "reward": round(reward, 2)
        })

        total_reward += reward

        state = next_state

        if done:
            break

    return path, total_reward, info


path, total_reward, info = evaluate_agent(
    env,
    agent
)


# ============================================================
# DISPLAY EVALUATION
# ============================================================

action_names = [
    "Keep Lane",
    "Change Left",
    "Change Right",
    "Accelerate",
    "Brake"
]

print("\nStep | Position | Lane | Speed | Action")

print("--------------------------------------------")

for item in path:

    print(
        f"{item['step']:4d} | "
        f"{item['position']:8.2f} | "
        f"{item['lane']:4d} | "
        f"{item['speed']:5.1f} | "
        f"{action_names[item['action']]}"
    )


print("\nTotal Evaluation Reward:", round(
    total_reward,
    2
))

print(
    "Final Position:",
    round(info["position"], 2)
)

print(
    "Final Lane:",
    info["lane"]
)

if info["collision"]:

    print("Result: COLLISION")

else:

    print("Result: DESTINATION REACHED")


# ============================================================
# REWARD GRAPH
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(reward_history)

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title(
    "PPO Training - Autonomous Lane Changing"
)

plt.grid(True)

plt.show()
