import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt


# ============================================================
# 1. GENERATE SYNTHETIC STOCK MARKET DATA
# ============================================================

np.random.seed(42)

days = 1000

prices = [100.0]

for i in range(days - 1):

    # Random market movement
    daily_return = np.random.normal(
        loc=0.0005,
        scale=0.02
    )

    new_price = prices[-1] * (1 + daily_return)

    prices.append(new_price)

prices = np.array(prices)


# ============================================================
# 2. TRADING ENVIRONMENT
# ============================================================

class TradingEnvironment:

    def __init__(self, prices):

        self.prices = prices

        # Actions
        # 0 = Sell
        # 1 = Hold
        # 2 = Buy

        self.action_size = 3

        # Starting capital
        self.initial_balance = 10000

        self.reset()

    def reset(self):

        self.day = 0

        self.balance = self.initial_balance

        self.shares = 0

        self.portfolio_value = self.initial_balance

        return self.get_state()

    # --------------------------------------------------------
    # Calculate state
    # --------------------------------------------------------

    def get_state(self):

        current_price = self.prices[self.day]

        # Price changes
        if self.day >= 5:

            returns = (
                self.prices[self.day - 4:self.day + 1]
                / self.prices[self.day - 5:self.day]
                - 1
            )

        else:

            returns = np.zeros(5)

        # Normalize position
        position_ratio = (
            self.shares * current_price
        ) / self.portfolio_value

        state = np.array([
            returns[0],
            returns[1],
            returns[2],
            returns[3],
            returns[4],
            position_ratio
        ], dtype=np.float32)

        return state

    # --------------------------------------------------------
    # Execute trading action
    # --------------------------------------------------------

    def step(self, action):

        current_price = self.prices[self.day]

        old_value = (
            self.balance +
            self.shares * current_price
        )

        # ----------------------------------------------------
        # BUY
        # ----------------------------------------------------

        if action == 2:

            # Use 10% of available balance
            amount = self.balance * 0.10

            shares_to_buy = int(
                amount / current_price
            )

            if shares_to_buy > 0:

                self.balance -= (
                    shares_to_buy * current_price
                )

                self.shares += shares_to_buy

        # ----------------------------------------------------
        # SELL
        # ----------------------------------------------------

        elif action == 0:

            # Sell 10% of holdings
            shares_to_sell = max(
                1,
                int(self.shares * 0.10)
            )

            shares_to_sell = min(
                shares_to_sell,
                self.shares
            )

            self.balance += (
                shares_to_sell * current_price
            )

            self.shares -= shares_to_sell

        # ----------------------------------------------------
        # HOLD
        # ----------------------------------------------------

        elif action == 1:

            pass

        # Move to next day
        self.day += 1

        done = (
            self.day >= len(self.prices) - 1
        )

        next_price = self.prices[self.day]

        # Current portfolio value
        self.portfolio_value = (
            self.balance +
            self.shares * next_price
        )

        # Portfolio return
        reward = (
            self.portfolio_value - old_value
        ) / old_value

        # ----------------------------------------------------
        # Risk penalty
        # ----------------------------------------------------

        # Penalize large negative daily returns
        if reward < -0.02:

            reward -= 0.01

        return (
            self.get_state(),
            reward,
            done,
            self.portfolio_value
        )


# ============================================================
# 3. POLICY NETWORK
# ============================================================

class PolicyNetwork(nn.Module):

    def __init__(self, state_size, action_size):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(state_size, 64),

            nn.ReLU(),

            nn.Linear(64, 64),

            nn.ReLU(),

            nn.Linear(64, action_size)
        )

    def forward(self, state):

        return self.network(state)


# ============================================================
# 4. REINFORCE AGENT
# ============================================================

class REINFORCEAgent:

    def __init__(
        self,
        state_size,
        action_size
    ):

        self.policy = PolicyNetwork(
            state_size,
            action_size
        )

        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=0.001
        )

        self.gamma = 0.99

    # --------------------------------------------------------
    # Select action
    # --------------------------------------------------------

    def select_action(self, state):

        state_tensor = torch.FloatTensor(
            state
        ).unsqueeze(0)

        logits = self.policy(
            state_tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        distribution = torch.distributions.Categorical(
            probabilities
        )

        action = distribution.sample()

        log_probability = distribution.log_prob(
            action
        )

        return (
            action.item(),
            log_probability
        )

    # --------------------------------------------------------
    # REINFORCE update
    # --------------------------------------------------------

    def update(
        self,
        log_probabilities,
        rewards
    ):

        returns = []

        discounted_return = 0

        # Calculate discounted returns
        for reward in reversed(rewards):

            discounted_return = (
                reward +
                self.gamma * discounted_return
            )

            returns.insert(
                0,
                discounted_return
            )

        returns = torch.tensor(
            returns,
            dtype=torch.float32
        )

        # Normalize returns
        returns = (
            returns - returns.mean()
        ) / (
            returns.std() + 1e-8
        )

        policy_loss = []

        for log_probability, G in zip(
            log_probabilities,
            returns
        ):

            policy_loss.append(
                -log_probability * G
            )

        loss = torch.stack(
            policy_loss
        ).sum()

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()


# ============================================================
# 5. TRAINING
# ============================================================

env = TradingEnvironment(prices)

agent = REINFORCEAgent(
    state_size=6,
    action_size=3
)

episodes = 100

reward_history = []

profit_history = []


print("============================================")
print(" REINFORCE AUTOMATED TRADING SYSTEM")
print("============================================")

print("\nTraining started...\n")


for episode in range(episodes):

    state = env.reset()

    log_probabilities = []

    rewards = []

    done = False

    while not done:

        # Select trading action
        action, log_probability = (
            agent.select_action(state)
        )

        # Execute trade
        next_state, reward, done, portfolio = (
            env.step(action)
        )

        log_probabilities.append(
            log_probability.squeeze()
        )

        rewards.append(reward)

        state = next_state

    # Update policy
    agent.update(
        log_probabilities,
        rewards
    )

    total_reward = sum(rewards)

    final_profit = (
        env.portfolio_value -
        env.initial_balance
    )

    reward_history.append(
        total_reward
    )

    profit_history.append(
        final_profit
    )

    if (episode + 1) % 10 == 0:

        print(
            f"Episode {episode + 1:3d} | "
            f"Reward: {total_reward:8.4f} | "
            f"Final Portfolio: "
            f"${env.portfolio_value:,.2f}"
        )


# ============================================================
# 6. EVALUATION
# ============================================================

print("\n============================================")
print(" EVALUATION")
print("============================================")

state = env.reset()

done = False

actions_taken = []

portfolio_values = [
    env.initial_balance
]

while not done:

    state_tensor = torch.FloatTensor(
        state
    ).unsqueeze(0)

    with torch.no_grad():

        logits = agent.policy(
            state_tensor
        )

        action = torch.argmax(
            logits,
            dim=1
        ).item()

    next_state, reward, done, portfolio = (
        env.step(action)
    )

    actions_taken.append(
        action
    )

    portfolio_values.append(
        portfolio
    )

    state = next_state


# ============================================================
# 7. PERFORMANCE METRICS
# ============================================================

initial_capital = env.initial_balance

final_capital = env.portfolio_value

profit = (
    final_capital -
    initial_capital
)

return_percentage = (
    profit /
    initial_capital
) * 100


# Maximum Drawdown

portfolio_values = np.array(
    portfolio_values
)

running_max = np.maximum.accumulate(
    portfolio_values
)

drawdown = (
    portfolio_values -
    running_max
) / running_max

max_drawdown = (
    np.min(drawdown) * 100
)


# Sharpe Ratio

daily_returns = (
    np.diff(portfolio_values)
    / portfolio_values[:-1]
)

if np.std(daily_returns) != 0:

    sharpe_ratio = (
        np.mean(daily_returns)
        / np.std(daily_returns)
    ) * np.sqrt(252)

else:

    sharpe_ratio = 0


# ============================================================
# 8. DISPLAY RESULTS
# ============================================================

print("\nInitial Capital :", 
      f"${initial_capital:,.2f}")

print("Final Capital   :", 
      f"${final_capital:,.2f}")

print("Profit/Loss     :", 
      f"${profit:,.2f}")

print("Return          :", 
      f"{return_percentage:.2f}%")

print("Maximum Drawdown:",
      f"{max_drawdown:.2f}%")

print("Sharpe Ratio    :",
      f"{sharpe_ratio:.2f}")


# ============================================================
# 9. ACTION STATISTICS
# ============================================================

buy_count = actions_taken.count(2)

hold_count = actions_taken.count(1)

sell_count = actions_taken.count(0)

print("\n============================================")
print(" TRADING ACTIONS")
print("============================================")

print("Buy Actions  :", buy_count)

print("Hold Actions :", hold_count)

print("Sell Actions :", sell_count)


# ============================================================
# 10. PLOT PORTFOLIO PERFORMANCE
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    portfolio_values,
    label="Portfolio Value"
)

plt.axhline(
    initial_capital,
    linestyle="--",
    label="Initial Capital"
)

plt.xlabel("Trading Day")

plt.ylabel("Portfolio Value")

plt.title(
    "REINFORCE Trading Agent Performance"
)

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# 11. TRAINING REWARD GRAPH
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    reward_history
)

plt.xlabel("Episode")

plt.ylabel("Total Reward")

plt.title(
    "REINFORCE Training Reward"
)

plt.grid(True)

plt.show()
