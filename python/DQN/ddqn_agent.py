import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
import random


# ============================================================
# Q-network
# ============================================================

def build_q_network(state_dim, n_actions, l2_strength=0.001):

    inputs = keras.Input(shape=(state_dim,), name="state")
    x = layers.Dense(256, activation="relu",
                      kernel_regularizer=keras.regularizers.l2(l2_strength))(inputs)
    x = layers.Dense(128, activation="relu",
                      kernel_regularizer=keras.regularizers.l2(l2_strength))(x)
    x = layers.Dense(64, activation="relu",
                      kernel_regularizer=keras.regularizers.l2(l2_strength))(x)
    outputs = layers.Dense(n_actions, activation="linear", name="q_values")(x)
    return keras.Model(inputs, outputs, name="dqn_q_network")


# ============================================================
# Uniform replay buffer
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity=50000, seed=None):
        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = self.rng.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ============================================================
# Prioritized replay buffer (optional, SumTree-based)
# ============================================================

class SumTree:
    """Minimal SumTree for proportional prioritized sampling."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.write_ptr = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write_ptr + self.capacity - 1
        self.data[self.write_ptr] = data
        self.update(idx, priority)
        self.write_ptr = (self.write_ptr + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    def __init__(self, capacity=50000, alpha=0.6711, eps=0.01, seed=None):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.eps = eps
        self.max_priority = 1.0
        self.rng = np.random.default_rng(seed)

    def push(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.tree.add(self.max_priority ** self.alpha, data)

    def sample(self, batch_size):
        idxs, priorities, batch = [], [], []
        segment = self.tree.total() / batch_size
        for i in range(batch_size):
            s = self.rng.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get(s)
            idxs.append(idx)
            priorities.append(p)
            batch.append(data)

        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            idxs,
        )

    def update_priorities(self, idxs, td_errors):
        for idx, err in zip(idxs, td_errors):
            priority = (abs(err) + self.eps) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.n_entries


# ============================================================
# DDQN Agent
# ============================================================

class DDQNAgent:
    def __init__(self, state_dim, n_actions,
                 learning_rate=0.00033,
                 gamma=0.95,
                 tau_soft_update=0.1,
                 l2_strength=0.00859,
                 use_per=False,
                 per_alpha=0.6711,
                 per_eps=0.01,
                 buffer_capacity=50000,
                 seed=None):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau_soft_update
        self.use_per = use_per

        self.online_net = build_q_network(state_dim, n_actions, l2_strength)
        self.target_net = build_q_network(state_dim, n_actions, l2_strength)
        self.target_net.set_weights(self.online_net.get_weights())

        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        if use_per:
            self.buffer = PrioritizedReplayBuffer(
                capacity=buffer_capacity, alpha=per_alpha, eps=per_eps, seed=seed)
        else:
            self.buffer = ReplayBuffer(capacity=buffer_capacity, seed=seed)

        self.rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    def act(self, state, epsilon):
        """Epsilon-greedy action selection."""
        if self.rng.uniform() < epsilon:
            return int(self.rng.integers(self.n_actions))
        q_values = self.online_net(state[None, :], training=False).numpy()[0]
        return int(np.argmax(q_values))

    def remember(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    # --------------------------------------------------------
    def _soft_update_target(self):
        online_weights = self.online_net.get_weights()
        target_weights = self.target_net.get_weights()
        new_weights = [
            self.tau * ow + (1.0 - self.tau) * tw
            for ow, tw in zip(online_weights, target_weights)
        ]
        self.target_net.set_weights(new_weights)

    @tf.function
    def _train_step(self, states, actions, rewards, next_states, dones):
        # Double-Q target: action selection from online net, evaluation
        # from target net.
        next_q_online = self.online_net(next_states, training=False)
        next_actions = tf.argmax(next_q_online, axis=1)

        next_q_target = self.target_net(next_states, training=False)
        next_actions_onehot = tf.one_hot(next_actions, self.n_actions)
        next_q_selected = tf.reduce_sum(next_q_target * next_actions_onehot, axis=1)

        targets = rewards + (1.0 - dones) * self.gamma * next_q_selected

        with tf.GradientTape() as tape:
            q_values = self.online_net(states, training=True)
            actions_onehot = tf.one_hot(actions, self.n_actions)
            q_selected = tf.reduce_sum(q_values * actions_onehot, axis=1)
            td_errors = targets - q_selected
            loss = tf.reduce_mean(tf.square(td_errors))
            loss += tf.reduce_sum(self.online_net.losses)  # L2 regularization

        grads = tape.gradient(loss, self.online_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.online_net.trainable_variables))

        return loss, td_errors

    def train_batch(self, batch_size=16, epochs=3):

        if len(self.buffer) < batch_size:
            return None

        losses = []
        for _ in range(epochs):
            if self.use_per:
                states, actions, rewards, next_states, dones, idxs = \
                    self.buffer.sample(batch_size)
            else:
                states, actions, rewards, next_states, dones = \
                    self.buffer.sample(batch_size)

            states_t = tf.convert_to_tensor(states)
            actions_t = tf.convert_to_tensor(actions)
            rewards_t = tf.convert_to_tensor(rewards)
            next_states_t = tf.convert_to_tensor(next_states)
            dones_t = tf.convert_to_tensor(dones)

            loss, td_errors = self._train_step(
                states_t, actions_t, rewards_t, next_states_t, dones_t)
            losses.append(float(loss))

            if self.use_per:
                self.buffer.update_priorities(idxs, td_errors.numpy())

        self._soft_update_target()
        return float(np.mean(losses))

    # --------------------------------------------------------
    def save(self, path):
        self.online_net.save(path)

    def load(self, path):
        self.online_net = keras.models.load_model(path)
        self.target_net.set_weights(self.online_net.get_weights())
