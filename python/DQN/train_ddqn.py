
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wind_turbine_env import WindTurbineEnv
from ddqn_agent import DDQNAgent


# ============================================================
# Hyperparameters (paper's "Arbitrary Set", Table 1)
# ============================================================
LEARNING_RATE = 0.01
N_EPISODES = 1000
PER_ALPHA = 0.75
PER_EPS = 0.01
BATCH_SIZE = 32
EPOCHS_PER_UPDATE = 3
TAU_SOFT_UPDATE = 0.1
EPSILON_GREEDY = 0.2          
GAMMA = 0.95
L2_STRENGTH = 0.001

USE_PER = False                
EPSILON_DECAY = False
EPSILON_START = 0.5
EPSILON_END = 0.05
EPSILON_DECAY_EPISODES = 200

MODEL_PATH = "../../modeli/turbina_vdc_w_ML.dmodl"  # same relative path style
                                                # as MPC.py
SAVE_PATH = "ddqn_pitch_model.keras"
SEED = 42


def get_epsilon(episode):
    if not EPSILON_DECAY:
        return EPSILON_GREEDY
    frac = min(episode / EPSILON_DECAY_EPISODES, 1.0)
    return EPSILON_START + frac * (EPSILON_END - EPSILON_START)


def run_episode(env, agent, epsilon, training=True):
    state = env.reset()
    episode_reward = 0.0
    episode_loss = []
    n_forbidden = 0
    done = False

    while not done:
        action = agent.act(state, epsilon if training else 0.0)
        next_state, reward, done, info = env.step(action)

        if training:
            agent.remember(state, action, reward, next_state, done)

        episode_reward += reward
        if info.get("forbidden", False):
            n_forbidden += 1

        state = next_state

    if training:
        loss = agent.train_batch(batch_size=BATCH_SIZE, epochs=EPOCHS_PER_UPDATE)
        if loss is not None:
            episode_loss.append(loss)

    return episode_reward, episode_loss, n_forbidden


def main():
    env = WindTurbineEnv(model_path=MODEL_PATH, seed=SEED)
    agent = DDQNAgent(
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        tau_soft_update=TAU_SOFT_UPDATE,
        l2_strength=L2_STRENGTH,
        use_per=USE_PER,
        per_alpha=PER_ALPHA,
        per_eps=PER_EPS,
        seed=SEED,
    )

    episode_rewards = []
    cumulative_rewards = []
    forbidden_counts = []
    running_cum = 0.0

    print(f"Starting DDQN training: {N_EPISODES} episodes, "
          f"state_dim={env.state_dim}, n_actions={env.n_actions}")

    for ep in range(N_EPISODES):
        epsilon = get_epsilon(ep)
        ep_reward, ep_loss, n_forbidden = run_episode(env, agent, epsilon, training=True)

        episode_rewards.append(ep_reward)
        running_cum += ep_reward
        cumulative_rewards.append(running_cum)
        forbidden_counts.append(n_forbidden)

        loss_str = f"{ep_loss[0]:.4f}" if ep_loss else "n/a"
        print(f"Episode {ep+1:4d}/{N_EPISODES} | "
              f"reward={ep_reward:8.2f} | "
              f"eps={epsilon:.3f} | "
              f"forbidden={n_forbidden:3d} | "
              f"loss={loss_str}")

        # Periodic checkpoint + quick greedy eval, every 25 episodes
        if (ep + 1) % 25 == 0:
            eval_reward, _, eval_forbidden = run_episode(
                env, agent, epsilon=0.0, training=False)
            print(f"  -> greedy eval: reward={eval_reward:.2f}, "
                  f"forbidden={eval_forbidden}")
            agent.save(SAVE_PATH)

    agent.save(SAVE_PATH)
    print(f"\nTraining complete. Model saved to {SAVE_PATH}")

    # --------------------------------------------------------
    # Training curves (mirrors paper's Figure 4a: reward + cumulative
    # reward over episodes)
    # --------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(episode_rewards, color="tab:blue", label="Episode reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Per-episode reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(cumulative_rewards, color="tab:orange", label="Cumulative reward")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Cumulative reward")
    axes[1].set_title("Cumulative reward over training")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("ddqn_training_curves.png", dpi=120)
    plt.close()

    np.savez(
        "ddqn_training_history.npz",
        episode_rewards=np.array(episode_rewards),
        cumulative_rewards=np.array(cumulative_rewards),
        forbidden_counts=np.array(forbidden_counts),
    )

    print("Saved: ddqn_training_curves.png, ddqn_training_history.npz")
    env.close()


if __name__ == "__main__":
    main()
