import argparse
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from breakout.config import DQNConfig, DATA_DIR, MODEL_DIR
from breakout.environment import make_breakout_env


class EvaluationCallback(BaseCallback):
    def __init__(self, eval_freq: int, n_eval_episodes: int, eval_env, seed: int):
        super().__init__()
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_env = eval_env
        self.seed = seed
        self.evaluation_steps = []
        self.evaluation_returns = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            returns = []
            for episode in range(self.n_eval_episodes):
                state, _ = self.eval_env.reset(seed=self.seed + 10000 + self.num_timesteps + episode)
                done = False
                total_reward = 0.0
                while not done:
                    # Valutazione deterministica della policy
                    action, _ = self.model.predict(state, deterministic=True)
                    state, reward, terminated, truncated, _ = self.eval_env.step(int(action))
                    total_reward += reward
                    done = terminated or truncated
                returns.append(total_reward)
            
            mean_return = float(np.mean(returns))
            self.evaluation_steps.append(self.num_timesteps)
            self.evaluation_returns.append(mean_return)
            print(f"[baseline] step={self.num_timesteps:>7} eval={mean_return:.2f}")
        return True


def run_baseline(steps: int, seed: int, device_name: str | None = None):
    env = make_breakout_env(seed)
    env = Monitor(env)
    
    eval_env = make_breakout_env(seed + 1000)
    
    config = DQNConfig(total_steps=steps)
    
    model = DQN(
        "CnnPolicy",
        env,
        learning_rate=config.learning_rate,
        buffer_size=config.replay_capacity,
        learning_starts=config.replay_start_size,
        batch_size=config.batch_size,
        tau=1.0,
        gamma=config.gamma,
        train_freq=config.train_frequency,
        target_update_interval=config.target_update_frequency,
        optimize_memory_usage=True,
        replay_buffer_kwargs={"handle_timeout_termination": False},
        exploration_initial_eps=config.epsilon_start,
        exploration_final_eps=config.epsilon_end,
        exploration_fraction=config.epsilon_decay_steps / steps,
        max_grad_norm=config.max_grad_norm,
        seed=seed,
        device=device_name or "auto"
    )
    
    eval_callback = EvaluationCallback(
        eval_freq=config.evaluation_frequency,
        n_eval_episodes=config.evaluation_episodes,
        eval_env=eval_env,
        seed=seed,
    )
    
    print(f"Inizio addestramento baseline (Stable-Baselines3) per {steps} passi...")
    model.learn(total_timesteps=steps, callback=eval_callback, log_interval=None)
    
    episode_returns = np.array(env.get_episode_rewards(), dtype=np.float32)
    episode_steps = np.array(env.get_episode_lengths(), dtype=np.int32)
    
    result_path = DATA_DIR / f"baseline_seed_{seed}.npz"
    np.savez_compressed(
        result_path,
        episode_returns=episode_returns,
        episode_steps=episode_steps,
        evaluation_steps=np.asarray(eval_callback.evaluation_steps, dtype=np.int32),
        evaluation_returns=np.asarray(eval_callback.evaluation_returns, dtype=np.float32),
    )
    
    model_path = MODEL_DIR / f"baseline_seed_{seed}"
    model.save(str(model_path))
    env.close()
    eval_env.close()
    
    print(f"[baseline] Risultati salvati in {result_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Esegue la baseline Stable-Baselines3 su Breakout")
    parser.add_argument("--steps", type=int, default=DQNConfig.total_steps)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cpu, cuda oppure auto")
    args = parser.parse_args()
    
    if args.steps < 1:
        raise ValueError("--steps deve essere positivo")
    run_baseline(steps=args.steps, seed=args.seed, device_name=args.device)


if __name__ == "__main__":
    main()
