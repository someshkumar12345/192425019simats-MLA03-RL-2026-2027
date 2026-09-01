import numpy as np
import random
import matplotlib.pyplot as plt

# ============================================================
# DYNAMIC PRICING USING MULTI-ARMED BANDITS
# ============================================================

np.random.seed(42)
random.seed(42)

# ------------------------------------------------------------
# Pricing options
# ------------------------------------------------------------

prices = np.array([
    50,
    60,
    70,
    80,
    90
])

num_prices = len(prices)

# Number of pricing decisions
num_rounds = 10000


# ============================================================
# TRUE DEMAND MODEL
# ============================================================

# Probability of a customer purchasing at each price.
#
# The retailer does NOT know these probabilities.
# They are used only by the simulator.

true_purchase_probability = np.array([
    0.90,   # Price = 50
    0.80,   # Price = 60
    0.65,   # Price = 70
    0.50,   # Price = 80
    0.35    # Price = 90
])


# ============================================================
# FUNCTION TO SIMULATE CUSTOMER PURCHASE
# ============================================================

def simulate_purchase(price_index):

    probability = true_purchase_probability[
        price_index
    ]

    purchase = (
        np.random.random() < probability
    )

    if purchase:
        return 1

    return 0


# ============================================================
# 1. EPSILON-GREEDY
# ============================================================

def epsilon_greedy(epsilon=0.1):

    # Estimated revenue for each price
    total_revenue = np.zeros(num_prices)

    # Number of times each price is selected
    counts = np.zeros(num_prices)

    cumulative_revenue = 0

    revenue_history = []

    for t in range(num_rounds):

        # ----------------------------------------------------
        # Explore or exploit
        # ----------------------------------------------------

        if random.random() < epsilon:

            # Exploration
            price_index = random.randint(
                0,
                num_prices - 1
            )

        else:

            # Exploitation
            average_revenue = np.divide(
                total_revenue,
                counts,
                out=np.zeros(num_prices),
                where=counts != 0
            )

            # Try an untested price first
            if np.any(counts == 0):

                untested = np.where(
                    counts == 0
                )[0]

                price_index = random.choice(
                    untested.tolist()
                )

            else:

                price_index = np.argmax(
                    average_revenue
                )

        # ----------------------------------------------------
        # Simulate customer
        # ----------------------------------------------------

        purchase = simulate_purchase(
            price_index
        )

        revenue = (
            prices[price_index]
            if purchase
            else 0
        )

        counts[price_index] += 1

        total_revenue[price_index] += revenue

        cumulative_revenue += revenue

        revenue_history.append(
            cumulative_revenue
        )

    return (
        revenue_history,
        counts,
        total_revenue
    )


# ============================================================
# 2. UCB - UPPER CONFIDENCE BOUND
# ============================================================

def ucb():

    total_revenue = np.zeros(num_prices)

    counts = np.zeros(num_prices)

    cumulative_revenue = 0

    revenue_history = []

    for t in range(num_rounds):

        # ----------------------------------------------------
        # Initially select every price once
        # ----------------------------------------------------

        if t < num_prices:

            price_index = t

        else:

            average_revenue = (
                total_revenue / counts
            )

            confidence = np.sqrt(
                (2 * np.log(t + 1))
                / counts
            )

            ucb_value = (
                average_revenue
                + confidence
            )

            price_index = np.argmax(
                ucb_value
            )

        # ----------------------------------------------------
        # Simulate purchase
        # ----------------------------------------------------

        purchase = simulate_purchase(
            price_index
        )

        revenue = (
            prices[price_index]
            if purchase
            else 0
        )

        counts[price_index] += 1

        total_revenue[price_index] += revenue

        cumulative_revenue += revenue

        revenue_history.append(
            cumulative_revenue
        )

    return (
        revenue_history,
        counts,
        total_revenue
    )


# ============================================================
# 3. THOMPSON SAMPLING
# ============================================================

def thompson_sampling():

    # Beta distribution parameters
    alpha = np.ones(num_prices)

    beta = np.ones(num_prices)

    cumulative_revenue = 0

    revenue_history = []

    for t in range(num_rounds):

        # ----------------------------------------------------
        # Sample purchase probability
        # ----------------------------------------------------

        sampled_probability = np.random.beta(
            alpha,
            beta
        )

        # Expected revenue = price × purchase probability
        sampled_revenue = (
            prices *
            sampled_probability
        )

        # Select price with highest sampled revenue
        price_index = np.argmax(
            sampled_revenue
        )

        # ----------------------------------------------------
        # Simulate purchase
        # ----------------------------------------------------

        purchase = simulate_purchase(
            price_index
        )

        revenue = (
            prices[price_index]
            if purchase
            else 0
        )

        # ----------------------------------------------------
        # Update Beta distribution
        # ----------------------------------------------------

        if purchase:

            alpha[price_index] += 1

        else:

            beta[price_index] += 1

        cumulative_revenue += revenue

        revenue_history.append(
            cumulative_revenue
        )

    return (
        revenue_history,
        alpha,
        beta
    )


# ============================================================
# RUN ALL ALGORITHMS
# ============================================================

epsilon_history, epsilon_counts, epsilon_revenue = (
    epsilon_greedy(epsilon=0.1)
)

ucb_history, ucb_counts, ucb_revenue = (
    ucb()
)

thompson_history, thompson_alpha, thompson_beta = (
    thompson_sampling()
)


# ============================================================
# FINAL REVENUE
# ============================================================

epsilon_final = epsilon_history[-1]
ucb_final = ucb_history[-1]
thompson_final = thompson_history[-1]


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("=" * 60)
print(" DYNAMIC PRICING USING MULTI-ARMED BANDITS")
print("=" * 60)

print("\nAvailable Prices:")

for i, price in enumerate(prices):

    print(
        f"Price {i + 1}: "
        f"${price}"
    )


print("\nTrue Purchase Probabilities:")

for i, probability in enumerate(
    true_purchase_probability
):

    print(
        f"${prices[i]} -> "
        f"{probability * 100:.1f}%"
    )


# ============================================================
# REVENUE COMPARISON
# ============================================================

print("\n" + "=" * 60)
print(" FINAL REVENUE")
print("=" * 60)

print(
    f"Epsilon-Greedy    : "
    f"${epsilon_final:,.2f}"
)

print(
    f"UCB               : "
    f"${ucb_final:,.2f}"
)

print(
    f"Thompson Sampling : "
    f"${thompson_final:,.2f}"
)


# ============================================================
# FIND BEST STRATEGY
# ============================================================

results = {

    "Epsilon-Greedy":
        epsilon_final,

    "UCB":
        ucb_final,

    "Thompson Sampling":
        thompson_final
}

best_strategy = max(
    results,
    key=results.get
)

print("\n" + "=" * 60)
print(" BEST PRICING STRATEGY")
print("=" * 60)

print(
    "Best Strategy:",
    best_strategy
)

print(
    f"Maximum Revenue: "
    f"${results[best_strategy]:,.2f}"
)


# ============================================================
# MOST SELECTED PRICES
# ============================================================

print("\n" + "=" * 60)
print(" PRICE SELECTION")
print("=" * 60)

epsilon_best_price = prices[
    np.argmax(epsilon_counts)
]

ucb_best_price = prices[
    np.argmax(ucb_counts)
]

thompson_best_price = prices[
    np.argmax(thompson_alpha - 1)
]


print(
    "Epsilon-Greedy selected most:",
    f"${epsilon_best_price}"
)

print(
    "UCB selected most:",
    f"${ucb_best_price}"
)

print(
    "Thompson Sampling selected most:",
    f"${thompson_best_price}"
)


# ============================================================
# REVENUE PER DECISION
# ============================================================

print("\n" + "=" * 60)
print(" AVERAGE REVENUE PER DECISION")
print("=" * 60)

print(
    "Epsilon-Greedy:",
    f"${epsilon_final / num_rounds:.2f}"
)

print(
    "UCB:",
    f"${ucb_final / num_rounds:.2f}"
)

print(
    "Thompson Sampling:",
    f"${thompson_final / num_rounds:.2f}"
)


# ============================================================
# GRAPH 1: CUMULATIVE REVENUE
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    epsilon_history,
    label="Epsilon-Greedy"
)

plt.plot(
    ucb_history,
    label="UCB"
)

plt.plot(
    thompson_history,
    label="Thompson Sampling"
)

plt.xlabel("Pricing Decisions")

plt.ylabel("Cumulative Revenue ($)")

plt.title(
    "Dynamic Pricing: Cumulative Revenue Comparison"
)

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# GRAPH 2: FINAL REVENUE COMPARISON
# ============================================================

algorithm_names = [
    "Epsilon-Greedy",
    "UCB",
    "Thompson Sampling"
]

revenues = [
    epsilon_final,
    ucb_final,
    thompson_final
]

plt.figure(figsize=(8, 5))

plt.bar(
    algorithm_names,
    revenues
)

plt.ylabel("Total Revenue ($)")

plt.title(
    "Final Revenue Comparison"
)

plt.grid(
    axis="y"
)

plt.show()
