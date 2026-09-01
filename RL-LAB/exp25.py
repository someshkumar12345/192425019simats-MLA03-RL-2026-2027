import numpy as np
import random
import matplotlib.pyplot as plt

# ============================================================
# ONLINE ADVERTISEMENT SELECTION USING BANDIT ALGORITHMS
# ============================================================

np.random.seed(42)
random.seed(42)

# ------------------------------------------------------------
# Advertisement configuration
# ------------------------------------------------------------

num_ads = 5
num_rounds = 10000

# True probability that each advertisement is clicked
# The algorithm does NOT know these values.
true_ctr = np.array([
    0.05,
    0.10,
    0.15,
    0.08,
    0.20
])


# ============================================================
# 1. EPSILON-GREEDY
# ============================================================

def epsilon_greedy(epsilon=0.1):

    clicks = np.zeros(num_ads)
    impressions = np.zeros(num_ads)

    cumulative_clicks = 0
    ctr_history = []

    for t in range(num_rounds):

        # ----------------------------------------------------
        # Exploration vs exploitation
        # ----------------------------------------------------

        if random.random() < epsilon:

            # Explore
            ad = random.randint(0, num_ads - 1)

        else:

            # Exploit
            estimated_ctr = np.divide(
                clicks,
                impressions,
                out=np.zeros(num_ads),
                where=impressions != 0
            )

            # If an ad has never been selected,
            # give it a chance to be explored.
            if np.any(impressions == 0):

                untested = np.where(
                    impressions == 0
                )[0]

                ad = random.choice(
                    untested.tolist()
                )

            else:

                ad = np.argmax(
                    estimated_ctr
                )

        # ----------------------------------------------------
        # Simulate user click
        # ----------------------------------------------------

        click = (
            np.random.random()
            < true_ctr[ad]
        )

        impressions[ad] += 1

        if click:

            clicks[ad] += 1
            cumulative_clicks += 1

        # Cumulative CTR
        ctr_history.append(
            cumulative_clicks / (t + 1)
        )

    return (
        ctr_history,
        clicks,
        impressions
    )


# ============================================================
# 2. UCB - UPPER CONFIDENCE BOUND
# ============================================================

def ucb():

    clicks = np.zeros(num_ads)
    impressions = np.zeros(num_ads)

    cumulative_clicks = 0
    ctr_history = []

    for t in range(num_rounds):

        # ----------------------------------------------------
        # Select each advertisement once initially
        # ----------------------------------------------------

        if t < num_ads:

            ad = t

        else:

            estimated_ctr = (
                clicks / impressions
            )

            confidence = np.sqrt(
                (2 * np.log(t + 1))
                / impressions
            )

            ucb_values = (
                estimated_ctr
                + confidence
            )

            ad = np.argmax(
                ucb_values
            )

        # ----------------------------------------------------
        # Simulate click
        # ----------------------------------------------------

        click = (
            np.random.random()
            < true_ctr[ad]
        )

        impressions[ad] += 1

        if click:

            clicks[ad] += 1
            cumulative_clicks += 1

        ctr_history.append(
            cumulative_clicks / (t + 1)
        )

    return (
        ctr_history,
        clicks,
        impressions
    )


# ============================================================
# 3. THOMPSON SAMPLING
# ============================================================

def thompson_sampling():

    # Beta distribution parameters
    alpha = np.ones(num_ads)
    beta = np.ones(num_ads)

    cumulative_clicks = 0
    ctr_history = []

    for t in range(num_rounds):

        # ----------------------------------------------------
        # Sample expected CTR from Beta distribution
        # ----------------------------------------------------

        sampled_ctr = np.random.beta(
            alpha,
            beta
        )

        ad = np.argmax(
            sampled_ctr
        )

        # ----------------------------------------------------
        # Simulate click
        # ----------------------------------------------------

        click = (
            np.random.random()
            < true_ctr[ad]
        )

        if click:

            alpha[ad] += 1
            cumulative_clicks += 1

        else:

            beta[ad] += 1

        ctr_history.append(
            cumulative_clicks / (t + 1)
        )

    return (
        ctr_history,
        alpha,
        beta
    )


# ============================================================
# RUN ALL THREE ALGORITHMS
# ============================================================

epsilon_history, epsilon_clicks, epsilon_impressions = (
    epsilon_greedy(epsilon=0.1)
)

ucb_history, ucb_clicks, ucb_impressions = (
    ucb()
)

thompson_history, thompson_alpha, thompson_beta = (
    thompson_sampling()
)


# ============================================================
# FINAL CTR CALCULATION
# ============================================================

epsilon_final_ctr = epsilon_history[-1]
ucb_final_ctr = ucb_history[-1]
thompson_final_ctr = thompson_history[-1]


print("=" * 60)
print(" ONLINE ADVERTISEMENT BANDIT EXPERIMENT")
print("=" * 60)

print("\nTrue CTR of advertisements:")
for i, ctr in enumerate(true_ctr):
    print(
        f"Advertisement {i + 1}: "
        f"{ctr * 100:.2f}%"
    )


print("\n" + "=" * 60)
print(" FINAL CLICK-THROUGH RATES")
print("=" * 60)

print(
    f"Epsilon-Greedy     : "
    f"{epsilon_final_ctr * 100:.2f}%"
)

print(
    f"UCB                : "
    f"{ucb_final_ctr * 100:.2f}%"
)

print(
    f"Thompson Sampling  : "
    f"{thompson_final_ctr * 100:.2f}%"
)


# ============================================================
# DETERMINE BEST ALGORITHM
# ============================================================

results = {
    "Epsilon-Greedy": epsilon_final_ctr,
    "UCB": ucb_final_ctr,
    "Thompson Sampling": thompson_final_ctr
}

best_algorithm = max(
    results,
    key=results.get
)

print("\n" + "=" * 60)
print(" BEST ALGORITHM")
print("=" * 60)

print(
    "Highest CTR:",
    best_algorithm
)

print(
    f"CTR: {results[best_algorithm] * 100:.2f}%"
)


# ============================================================
# MOST SELECTED ADVERTISEMENT
# ============================================================

epsilon_best_ad = np.argmax(
    epsilon_clicks
)

ucb_best_ad = np.argmax(
    ucb_clicks
)

thompson_best_ad = np.argmax(
    thompson_alpha - 1
)

print("\n" + "=" * 60)
print(" MOST SELECTED / LEARNED AD")
print("=" * 60)

print(
    "Epsilon-Greedy:",
    epsilon_best_ad + 1
)

print(
    "UCB:",
    ucb_best_ad + 1
)

print(
    "Thompson Sampling:",
    thompson_best_ad + 1
)


# ============================================================
# PLOT CTR OVER TIME
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

plt.xlabel("Number of Advertisement Impressions")

plt.ylabel("Cumulative Click-Through Rate")

plt.title(
    "Comparison of Bandit Algorithms for Online Advertisement Selection"
)

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# FINAL BAR CHART
# ============================================================

algorithms = [
    "Epsilon-Greedy",
    "UCB",
    "Thompson Sampling"
]

final_ctrs = [
    epsilon_final_ctr,
    ucb_final_ctr,
    thompson_final_ctr
]

plt.figure(figsize=(8, 5))

plt.bar(
    algorithms,
    final_ctrs
)

plt.ylabel("Final CTR")

plt.title(
    "Final CTR Comparison"
)

plt.grid(
    axis="y"
)

plt.show()
