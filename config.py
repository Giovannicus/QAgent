"""All the settings you're likely to want to tweak, in one place.
Change values here instead of digging through qlearning_agent.py."""

# Which MiniGrid environment to train/watch on.
# "MiniGrid-Empty-16x16-v0" is a static, empty room: easy, good for
# checking the agent works at all.
# "MiniGrid-Dynamic-Obstacles-16x16-v0" adds moving obstacles: much
# harder for a tabular agent, see project notes on why.
ENV_ID = "MiniGrid-Empty-16x16-v0"

# How many episodes to train for, and how often to print progress.
# 1000 gives a comfortable safety margin: with the full-grid observation
# (see make_env below) the greedy policy already solves the env reliably
# (30/30 test seeds) after just 200 episodes.
NUM_EPISODES = 1000
LOG_EVERY = 100

# Where to save/load the trained Q-table.
Q_TABLE_PATH = "q_table.pkl"

# How many fresh episodes to play in the "watch" window.
NUM_WATCH_EPISODES = 5

# MiniGrid numbers "turn left"=0, "turn right"=1, "move forward"=2 the
# same way in every environment; the remaining actions (pickup, drop,
# toggle, done) are irrelevant for pure navigation tasks like the ones
# used here and would only waste exploration budget during training.
NUM_ACTIONS = 3

# Small chance of a random move even while "watching" (exploring=False
# everywhere else). Pure greedy playback can get stuck bouncing between
# two states forever (e.g. turn right, turn left, turn right, ...) if
# their Q-values are nearly tied; a little randomness lets it escape.
WATCH_EXPLORATION_RATE = 0.05

# --- Q-learning hyperparameters ---

# alpha: how much weight a new experience gets vs. what's already learned.
# Higher = learns faster but noisier; lower = more stable but slower.
LEARNING_RATE = 0.1

# gamma: how much future rewards matter compared to immediate ones.
# Closer to 1 = plans further ahead; closer to 0 = only cares about the
# next step.
DISCOUNT_FACTOR = 0.99

# epsilon: starting probability of taking a random action instead of the
# best known one. Starts high so the agent explores before it has
# anything useful in the Q-table.
EXPLORATION_RATE = 0.9

# epsilon never goes below this, so the agent keeps exploring a little
# even late in training.
MIN_EXPLORATION_RATE = 0.05

# How much epsilon shrinks after every episode (multiplicative decay).
# Closer to 1 = explores for longer before settling on learned behavior.
EXPLORATION_DECAY = 0.9995
