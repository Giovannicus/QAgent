import gymnasium as gym
import minigrid  # noqa: F401  registers MiniGrid envs with gymnasium

env = gym.make("MiniGrid-Dynamic-Obstacles-16x16-v0")
obs, info = env.reset(seed=0)

print("observation space:", env.observation_space)
print("action space:", env.action_space)
print("initial obs keys:", obs.keys() if isinstance(obs, dict) else type(obs))

terminated = truncated = False
steps = 0
while not (terminated or truncated) and steps < 50:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    steps += 1

print(f"episode ended after {steps} steps, reward={reward}, terminated={terminated}, truncated={truncated}")
env.close()
