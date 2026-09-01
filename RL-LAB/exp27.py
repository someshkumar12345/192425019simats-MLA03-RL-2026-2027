import random
import matplotlib.pyplot as plt


# ============================================================
# AUTONOMOUS CAR ROAD NETWORK
# ============================================================

# Road network:
#
#       B -------- C
#       |          |
#       |          |
#       A -------- D -------- E
#
# Start = A
# Destination = E
#
# Each node represents an intersection.
# Each edge represents a road.

road_network = {
    "A": ["B", "D"],
    "B": ["A", "C"],
    "C": ["B", "D"],
    "D": ["A", "C", "E"],
    "E": ["D"]
}


# ============================================================
# ROAD CONDITIONS
# ============================================================

# Speed limits for roads

speed_limits = {
    ("A", "B"): 40,
    ("B", "A"): 40,

    ("B", "C"): 50,
    ("C", "B"): 50,

    ("C", "D"): 40,
    ("D", "C"): 40,

    ("A", "D"): 50,
    ("D", "A"): 50,

    ("D", "E"): 60,
    ("E", "D"): 60
}


# ============================================================
# TRAFFIC SIGNALS
# ============================================================

# True = green
# False = red

traffic_lights = {
    "A": True,
    "B": True,
    "C": False,
    "D": True,
    "E": True
}


# ============================================================
# AUTONOMOUS CAR
# ============================================================

class AutonomousCar:

    def __init__(self, policy):

        self.start = "A"
        self.destination = "E"

        self.position = self.start

        self.policy = policy

        self.path = [self.start]

        self.total_reward = 0

        self.total_distance = 0

        self.violations = 0

        self.steps = 0

        self.collisions = 0

    # --------------------------------------------------------
    # Check traffic rule
    # --------------------------------------------------------

    def check_traffic_rule(self, current, next_node):

        # Red signal means the car should wait.
        if not traffic_lights[current]:

            return False

        return True

    # --------------------------------------------------------
    # Calculate road distance
    # --------------------------------------------------------

    def road_distance(self, current, next_node):

        distances = {
            ("A", "B"): 5,
            ("B", "C"): 4,
            ("C", "D"): 6,
            ("A", "D"): 7,
            ("D", "E"): 5
        }

        if (current, next_node) in distances:

            return distances[
                (current, next_node)
            ]

        return distances[
            (next_node, current)
        ]

    # --------------------------------------------------------
    # Select next road using policy
    # --------------------------------------------------------

    def choose_next_node(self):

        possible_nodes = road_network[
            self.position
        ]

        if self.policy == "safe":

            return self.safe_policy(
                possible_nodes
            )

        elif self.policy == "fast":

            return self.fast_policy(
                possible_nodes
            )

        elif self.policy == "balanced":

            return self.balanced_policy(
                possible_nodes
            )

        else:

            return random.choice(
                possible_nodes
            )

    # --------------------------------------------------------
    # Safe policy
    # --------------------------------------------------------

    def safe_policy(self, possible_nodes):

        # Prefer roads that move toward destination
        # while following traffic rules.

        if self.position == "A":

            return "D"

        if self.position == "B":

            return "C"

        if self.position == "C":

            return "D"

        if self.position == "D":

            return "E"

        return possible_nodes[0]

    # --------------------------------------------------------
    # Fast policy
    # --------------------------------------------------------

    def fast_policy(self, possible_nodes):

        # Choose road with highest speed limit.

        best_node = None
        best_speed = -1

        for node in possible_nodes:

            speed = speed_limits[
                (self.position, node)
            ]

            if speed > best_speed:

                best_speed = speed
                best_node = node

        return best_node

    # --------------------------------------------------------
    # Balanced policy
    # --------------------------------------------------------

    def balanced_policy(self, possible_nodes):

        best_node = None
        best_score = -999999

        for node in possible_nodes:

            distance = self.road_distance(
                self.position,
                node
            )

            speed = speed_limits[
                (self.position, node)
            ]

            # Higher speed and lower distance
            # produce a better score.

            score = (
                speed * 0.5
                - distance
            )

            # Penalize moving away from destination.

            if node == "A" and self.position != "A":
                score -= 20

            if node == "B":
                score -= 2

            if node == "C":
                score -= 1

            if node == "D":
                score += 5

            if node == "E":
                score += 100

            if score > best_score:

                best_score = score
                best_node = node

        return best_node

    # --------------------------------------------------------
    # Execute one driving step
    # --------------------------------------------------------

    def step(self):

        if self.position == self.destination:

            return True

        next_node = self.choose_next_node()

        # ----------------------------------------------------
        # Traffic rule checking
        # ----------------------------------------------------

        if not self.check_traffic_rule(
            self.position,
            next_node
        ):

            # Wait at red signal.
            self.total_reward -= 2

            print(
                f"Car waiting at red signal "
                f"at {self.position}"
            )

            traffic_lights[
                self.position
            ] = True

            return False

        # ----------------------------------------------------
        # Calculate road information
        # ----------------------------------------------------

        distance = self.road_distance(
            self.position,
            next_node
        )

        speed = speed_limits[
            (self.position, next_node)
        ]

        # ----------------------------------------------------
        # Reward
        # ----------------------------------------------------

        reward = 10

        # Smaller distance is preferred.
        reward -= distance * 0.5

        # Following speed limits.
        reward += speed * 0.05

        # ----------------------------------------------------
        # Move vehicle
        # ----------------------------------------------------

        print(
            f"Car moves: "
            f"{self.position} -> {next_node}"
        )

        self.position = next_node

        self.path.append(
            next_node
        )

        self.total_distance += distance

        self.total_reward += reward

        self.steps += 1

        # ----------------------------------------------------
        # Destination reached
        # ----------------------------------------------------

        if self.position == self.destination:

            self.total_reward += 100

            return True

        return False

    # --------------------------------------------------------
    # Run simulation
    # --------------------------------------------------------

    def run(self, max_steps=20):

        print(
            f"\nPolicy: {self.policy}"
        )

        print(
            "Start:",
            self.start
        )

        print(
            "Destination:",
            self.destination
        )

        for _ in range(max_steps):

            finished = self.step()

            if finished:

                break

        print(
            "Path:",
            " -> ".join(self.path)
        )

        print(
            "Total Distance:",
            self.total_distance
        )

        print(
            "Total Reward:",
            round(
                self.total_reward,
                2
            )
        )

        print(
            "Traffic Violations:",
            self.violations
        )

        print(
            "Collisions:",
            self.collisions
        )

        return {
            "policy": self.policy,
            "success":
                self.position == self.destination,
            "distance":
                self.total_distance,
            "reward":
                self.total_reward,
            "violations":
                self.violations,
            "collisions":
                self.collisions,
            "steps":
                self.steps
        }


# ============================================================
# EVALUATE POLICIES
# ============================================================

policies = [
    "safe",
    "fast",
    "balanced"
]

results = []


print("=" * 60)
print(" AUTONOMOUS CAR POLICY EVALUATION")
print("=" * 60)


for policy in policies:

    # Reset traffic lights
    traffic_lights = {
        "A": True,
        "B": True,
        "C": False,
        "D": True,
        "E": True
    }

    car = AutonomousCar(
        policy
    )

    result = car.run()

    results.append(
        result
    )


# ============================================================
# DISPLAY COMPARISON
# ============================================================

print("\n")
print("=" * 70)
print(" POLICY COMPARISON")
print("=" * 70)

print(
    f"{'Policy':<12}"
    f"{'Success':<10}"
    f"{'Distance':<12}"
    f"{'Reward':<12}"
    f"{'Violations':<12}"
)

print("-" * 70)

for result in results:

    print(
        f"{result['policy']:<12}"
        f"{str(result['success']):<10}"
        f"{result['distance']:<12}"
        f"{result['reward']:<12.2f}"
        f"{result['violations']:<12}"
    )


# ============================================================
# DETERMINE BEST POLICY
# ============================================================

successful_results = [
    result
    for result in results
    if result["success"]
]

if successful_results:

    best_policy = max(
        successful_results,
        key=lambda x: x["reward"]
    )

    print("\n")
    print("=" * 60)
    print(" BEST POLICY")
    print("=" * 60)

    print(
        "Policy:",
        best_policy["policy"]
    )

    print(
        "Reward:",
        round(
            best_policy["reward"],
            2
        )
    )

    print(
        "Distance:",
        best_policy["distance"]
    )

    print(
        "Destination Reached:",
        best_policy["success"]
    )


# ============================================================
# VISUALIZE POLICY PERFORMANCE
# ============================================================

policy_names = [
    result["policy"]
    for result in results
]

rewards = [
    result["reward"]
    for result in results
]

distances = [
    result["distance"]
    for result in results
]


# ------------------------------------------------------------
# Reward graph
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    policy_names,
    rewards
)

plt.xlabel(
    "Driving Policy"
)

plt.ylabel(
    "Total Reward"
)

plt.title(
    "Autonomous Car Policy Reward Comparison"
)

plt.grid(
    axis="y"
)

plt.show()


# ------------------------------------------------------------
# Distance graph
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    policy_names,
    distances
)

plt.xlabel(
    "Driving Policy"
)

plt.ylabel(
    "Total Distance"
)

plt.title(
    "Autonomous Car Distance Comparison"
)

plt.grid(
    axis="y"
)

plt.show()
