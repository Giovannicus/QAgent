import pickle

import gymnasium as gym
import minigrid  # noqa: F401  registers MiniGrid environments with gymnasium
from minigrid.wrappers import FullyObsWrapper
import numpy as np
from collections import defaultdict

import config


def make_env(env_id, render_mode=None):
    """Create the environment with a FULL view of the grid instead of the
    default partial 7x7 view centered on the agent.
    Why this matters: with the partial view, most cells in an empty room
    look identical (just open floor in every direction), so two different
    real positions can produce the exact same observation. The agent then
    can't tell them apart and the Q-table can't learn a consistent action
    for that "state" - it's really two different states colliding into
    one. The full view includes the agent's absolute position, so each
    real situation maps to its own distinct state."""
    return FullyObsWrapper(gym.make(env_id, render_mode=render_mode))


def encode_state(observation):
    """Turn the observation (full grid image + direction) into a hashable
    key so it can be used as an index into the Q-table.
    The theoretical state space is huge, but only a small fraction of
    states are ever actually visited: a dict is therefore much more
    efficient than a dense table."""
    image = observation["image"]
    direction = observation["direction"]
    return (image.tobytes(), int(direction))


class QLearningAgent:
    def __init__(
        self,
        num_actions,
        learning_rate=config.LEARNING_RATE,            # alpha: how much weight the new TD target gets vs. the current value
        discount_factor=config.DISCOUNT_FACTOR,         # gamma: how much future rewards matter relative to immediate ones
        exploration_rate=config.EXPLORATION_RATE,       # epsilon: probability of picking a random action (exploration)
        min_exploration_rate=config.MIN_EXPLORATION_RATE,
        exploration_decay=config.EXPLORATION_DECAY,
    ):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay = exploration_decay

        # Q-table: any state seen for the first time starts at 0 for every action
        self.q_table = defaultdict(lambda: np.zeros(num_actions))

    def choose_action(self, observation):
        """Epsilon-greedy strategy: with probability epsilon explore a
        random action, otherwise exploit the best known action."""
        explore = np.random.random() < self.exploration_rate
        if explore:
            return np.random.randint(self.num_actions)

        state = encode_state(observation)
        return int(np.argmax(self.q_table[state]))

    def learn(self, observation, action, reward, next_observation, episode_done):
        """Q-learning update (Bellman equation):
        Q(s,a) <- Q(s,a) + alpha * (target - Q(s,a))
        where target = reward + gamma * max_a' Q(s', a')
        If the episode has ended there is no "future", so the target is
        simply the reward just received."""
        state = encode_state(observation)
        next_state = encode_state(next_observation)

        best_future_value = 0.0 if episode_done else np.max(self.q_table[next_state])
        td_target = reward + self.discount_factor * best_future_value
        td_error = td_target - self.q_table[state][action]

        self.q_table[state][action] += self.learning_rate * td_error

    def decay_exploration(self):
        """Decay epsilon after each episode: explore a lot early on,
        exploit more and more of what's been learned over time."""
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay,
        )


def run_episode(env, agent, training=True):
    """Run one full episode. If training=True, the agent learns at every
    step; otherwise it only picks actions (evaluation mode)."""
    observation, _info = env.reset()
    episode_done = False
    total_reward = 0.0

    while not episode_done:
        action = agent.choose_action(observation)
        next_observation, reward, terminated, truncated, _info = env.step(action)
        episode_done = terminated or truncated

        if training:
            agent.learn(observation, action, reward, next_observation, episode_done)

        observation = next_observation
        total_reward += reward

    return total_reward


def train(env_id=config.ENV_ID, num_episodes=config.NUM_EPISODES, log_every=config.LOG_EVERY):
    env = make_env(env_id)
    agent = QLearningAgent(num_actions=config.NUM_ACTIONS)

    reward_history = []
    for episode in range(1, num_episodes + 1):
        episode_reward = run_episode(env, agent, training=True)
        agent.decay_exploration()
        reward_history.append(episode_reward)

        if episode % log_every == 0:
            recent_avg_reward = np.mean(reward_history[-log_every:])
            print(
                f"episode {episode:5d} | "
                f"avg reward (last {log_every})={recent_avg_reward:.3f} | "
                f"epsilon={agent.exploration_rate:.3f} | "
                f"states seen={len(agent.q_table)}"
            )

    env.close()
    return agent, reward_history


def save_q_table(agent, path=config.Q_TABLE_PATH):
    """Save what the agent has learned so far to disk. This is separate
    from training on purpose: train once, then watch/reload as many times
    as you want without paying the training time again."""
    with open(path, "wb") as f:
        pickle.dump(dict(agent.q_table), f)


def load_q_table(agent, path=config.Q_TABLE_PATH):
    """Load a previously saved Q-table into an (untrained) agent with the
    same number of actions."""
    with open(path, "rb") as f:
        saved_q_table = pickle.load(f)
    agent.q_table.update(saved_q_table)


def watch_agent(agent, env_id=config.ENV_ID, num_episodes=config.NUM_WATCH_EPISODES):
    """Open a window and play a few FRESH episodes (not a replay of
    training) so you can see what the agent learned. render_mode="human"
    is enough: gymnasium/MiniGrid draw the grid automatically after every
    reset/step, no extra code needed.
    Exploration is set to a small value (not exactly 0) here: a purely
    greedy agent can get stuck bouncing forever between two states with
    near-tied Q-values (e.g. turn right, turn left, turn right, ...); a
    little randomness lets it break out of that kind of loop."""
    env = make_env(env_id, render_mode="human")

    original_exploration_rate = agent.exploration_rate
    agent.exploration_rate = config.WATCH_EXPLORATION_RATE
    try:
        for episode in range(1, num_episodes + 1):
            episode_reward = run_episode(env, agent, training=False)
            print(f"Episodio {episode}/{num_episodes}: reward={episode_reward:.3f}")
    finally:
        agent.exploration_rate = original_exploration_rate
        env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a Q-learning MiniGrid agent, or watch a previously trained one play."
    )
    parser.add_argument(
        "mode",
        choices=["train", "watch"],
        help=(
            "'train': run training (no window, prints progress) and save the result to disk. "
            "'watch': load a previously saved agent and open a window to see it play, instantly."
        ),
    )
    parser.add_argument("--env-id", default=config.ENV_ID)
    parser.add_argument("--episodes", type=int, default=config.NUM_EPISODES, help="only used by 'train'")
    parser.add_argument("--watch-episodes", type=int, default=config.NUM_WATCH_EPISODES, help="only used by 'watch'")
    parser.add_argument("--q-table-path", default=config.Q_TABLE_PATH)
    args = parser.parse_args()

    if args.mode == "train":
        trained_agent, _reward_history = train(env_id=args.env_id, num_episodes=args.episodes)
        save_q_table(trained_agent, args.q_table_path)
        print(f"Q-table saved to {args.q_table_path}")
    else:
        agent = QLearningAgent(num_actions=config.NUM_ACTIONS)

        load_q_table(agent, args.q_table_path)
        watch_agent(agent, env_id=args.env_id, num_episodes=args.watch_episodes)
