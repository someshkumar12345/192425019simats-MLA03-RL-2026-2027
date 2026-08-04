import gymnasium as gym

env = gym.make("FrozenLake-v1")

state, info = env.reset()
done = False
total_reward = 0

while not done:
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward

print("Episode Reward:", total_reward)

env.close()
