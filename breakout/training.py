"""Training, valutazione e persistenza dei risultati DQN."""

import random
from dataclasses import asdict, replace

import numpy as np
import torch

from breakout.agent import DQNAgent
from breakout.config import DATA_DIR, MODEL_DIR, DQNConfig
from breakout.environment import make_breakout_env
from breakout.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, TransitionBatch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(agent: DQNAgent, seed: int, episodes: int, random_policy: bool = False) -> float:
    env = make_breakout_env(seed)
    returns = []
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        done, total_reward = False, 0.0
        while not done:
            action = env.action_space.sample() if random_policy else agent.select_action(state, 0.0)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        returns.append(total_reward)
    env.close()
    return float(np.mean(returns))


def run_training(name: str, config: DQNConfig, seed: int = 42, device_name: str | None = None):
    """Esegue un esperimento DQN e salva metriche e pesi della rete online."""
    set_seed(seed)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Initialize environment, agent, and replay memory buffer
    env = make_breakout_env(seed)
    state, _ = env.reset(seed=seed)
    agent = DQNAgent(env.action_space.n, config, device)
    
    if config.use_per:
        replay = PrioritizedReplayBuffer(config.replay_capacity, tuple(np.asarray(state).shape), alpha=config.per_alpha)
    else:
        replay = ReplayBuffer(config.replay_capacity, tuple(np.asarray(state).shape))

    # Tracking metrics for analysis and plotting
    episode_returns, episode_steps, losses = [], [], []
    evaluation_steps, evaluation_returns = [], []
    current_return, current_length = 0.0, 0

    for step in range(1, config.total_steps + 1):
        action = agent.select_action(state, config.epsilon(step))
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Clip rewards to [-1, 1] for gradient stability
        clipped_reward = float(np.clip(reward, -1.0, 1.0))
        current_return += reward
        current_length += 1

        if config.use_replay:
            # Store transition in buffer and sample mini-batches
            replay.add(state, action, clipped_reward, next_state, done)
            if (hasattr(replay, 'tree') and replay.tree.size >= config.replay_start_size) or \
               (not hasattr(replay, 'tree') and replay.size >= config.replay_start_size):
                if step % config.train_frequency == 0:
                    if config.use_per:
                        fraction = min(step / config.total_steps, 1.0)
                        beta = config.per_beta_start + fraction * (1.0 - config.per_beta_start)
                        batch = replay.sample(config.batch_size, device, beta=beta)
                    else:
                        batch = replay.sample(config.batch_size, device)
                        
                    loss, td_errors = agent.update(batch, weights=batch.weights)
                    losses.append(loss)
                    
                    if config.use_per:
                        replay.update_priorities(batch.indices, td_errors)
        elif step % config.train_frequency == 0:
            # Replay ablation study: perform immediate update on single trajectory step
            batch = TransitionBatch(
                torch.as_tensor(np.asarray(state)[None], device=device),
                torch.tensor([action], device=device),
                torch.tensor([clipped_reward], device=device),
                torch.as_tensor(np.asarray(next_state)[None], device=device),
                torch.tensor([float(done)], device=device),
            )
            loss, _ = agent.update(batch)
            losses.append(loss)

        state = next_state
        if done:
            # Record episode metrics and reset environment for next episode
            episode_returns.append(current_return)
            episode_steps.append(current_length)
            state, _ = env.reset()
            current_return, current_length = 0.0, 0

        # Periodically hard-update target network weights to stabilize Q-value targets
        if config.use_target_network and step % config.target_update_frequency == 0:
            agent.update_target_network()

        # Periodically evaluate policy performance without exploration noise
        if step % config.evaluation_frequency == 0:
            score = evaluate(agent, seed + 10_000 + step, config.evaluation_episodes)
            evaluation_steps.append(step)
            evaluation_returns.append(score)
            print(f"[{name}] step={step:>7} epsilon={config.epsilon(step):.3f} eval={score:.2f}")

    env.close()
    result_path = DATA_DIR / f"{name}_seed_{seed}.npz"
    np.savez_compressed(
        result_path,
        episode_returns=np.asarray(episode_returns, dtype=np.float32),
        episode_steps=np.asarray(episode_steps, dtype=np.int32),
        losses=np.asarray(losses, dtype=np.float32),
        evaluation_steps=np.asarray(evaluation_steps, dtype=np.int32),
        evaluation_returns=np.asarray(evaluation_returns, dtype=np.float32),
        random_baseline=evaluate(agent, seed + 20_000, config.evaluation_episodes, random_policy=True),
        config=np.array(asdict(config), dtype=object),
    )

    # Save final model weights
    torch.save(agent.online_net.state_dict(), MODEL_DIR / f"{name}_seed_{seed}.pt")
    print(f"[{name}] risultati salvati in {result_path}")
    return result_path


def ablation_config(base: DQNConfig, *, replay: bool, target: bool) -> DQNConfig:
    """Helper to toggle experience replay and target network for ablation studies"""
    return replace(base, use_replay=replay, use_target_network=target)
