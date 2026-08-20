"""Creazione e preprocessing standard dell'ambiente Atari Breakout."""

import ale_py
import gymnasium as gym

gym.register_envs(ale_py)

from breakout.config import ENV_ID


def make_breakout_env(seed: int | None = None, render_mode: str | None = None):
    """Crea Breakout con grayscale 84x84, frame skip 4 e 4 frame stacked."""
    env = gym.make(ENV_ID, frameskip=1, repeat_action_probability=0.0, render_mode=render_mode)
    env = gym.wrappers.AtariPreprocessing(
        env, noop_max=30, frame_skip=4, screen_size=84,
        terminal_on_life_loss=False, grayscale_obs=True,
        grayscale_newaxis=False, scale_obs=False,
    )
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env
