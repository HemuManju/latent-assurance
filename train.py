import gymnasium as gym

from utils import skip_run

with skip_run("skip", "simulator") as check, check():
    env = gym.make("CarRacing-v3", render_mode="human")
    observation, info = env.reset(seed=42)
    env.render()
    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            observation, info = env.reset()
    env.close()
