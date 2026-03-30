"""
Atari DQN — MsPacman-v5 with Stable-Baselines3
================================================
This script trains a Deep Q-Network (DQN) agent on ALE/MsPacman-v5 and lets
you watch it play.  It is adapted from the Breakout base template provided
by the course.

Why MsPacman is harder than Breakout
-------------------------------------
  * Action space: 9 discrete actions vs. 4 in Breakout.
  * Moving objects: 4 ghosts with non-trivial AI that change behaviour after
    a power pellet (scatter → chase and back).
  * Partial observability: the agent cannot see around corners; a ghost can
    appear suddenly, requiring reactive planning.
  * Reward structure: eating pellets (+10 each), power pellets (+50), ghosts
    (+200/+400/+800/+1600 in sequence), fruit bonus (+100 to +5000).

How DQN handles it (high level)
---------------------------------
  1. Raw 210×160 RGB frames are preprocessed to 84×84 grayscale.
  2. The last N_STACK=4 frames are stacked so the CNN sees motion (ghost
     direction, Pac-Man speed).
  3. The CNN maps each stacked frame to a Q-value per action.
  4. An ε-greedy schedule drives exploration of the maze early on, then
     shifts to exploitation once the policy matures.
  5. A replay buffer decouples data collection from learning; a target
     network provides a stable regression target.

Hyperparameter differences from the Breakout defaults
-------------------------------------------------------
  buffer_size       100_000  — larger maze state space needs a richer buffer
  learning_starts    20_000  — more warm-up so the buffer samples diverse maze
                               states before training begins
  exploration_fraction 0.20  — 20% of steps for ε-decay; maze navigation
                               requires longer exploration than Breakout
  target_update_interval 2_000 — less frequent sync stabilises learning against
                               the noisy, multi-objective reward signal
  All other defaults match the Breakout template (lr=1e-4, γ=0.99, etc.).

Usage
-----
  # Train with built-in defaults  (saves to models/mspacman_dqn.zip)
  python mspacman_dqn.py --mode train --model-path models/mspacman_dqn

  # Train a named experiment from the sweep file
  python mspacman_dqn.py --mode train --experiment exp_02_lr_high \\
      --model-path models/mspacman_exp02

  # Watch the trained agent play
  python mspacman_dqn.py --mode play --model-path models/mspacman_dqn --episodes 3

  # Run all experiments from sweep_configs_mspacman.json
  python mspacman_dqn.py --mode sweep --sweep-file sweep_configs_mspacman.json

  # Monitor training in TensorBoard
  python -m tensorboard.main --logdir logs/mspacman_dqn --port 6006
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import ale_py
import gymnasium as gym

gym.register_envs(ale_py)  # register ALE environments in the gymnasium namespace

from torch.utils.tensorboard import SummaryWriter

from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# Target game — Group 7 assignment
ENV_ID = "ALE/MsPacman-v5"

# Number of consecutive frames stacked together as a single observation.
# Stacking gives the agent a sense of motion (ghost direction, Pac-Man velocity).
N_STACK = 4


# ── TensorBoard Callback ───────────────────────────────────────────────────────

class TensorBoardCallback(BaseCallback):
    """Custom callback that logs per-episode metrics to TensorBoard.

    Reuses SB3's internal SummaryWriter so all scalars land in the same
    event file as the built-in rollout/ and train/ metrics.

    Scalars logged by this callback:
      - training/episode_reward : total reward accumulated in each episode
      - training/epsilon        : current exploration rate (ε), logged every step

    SB3 built-in scalars (also visible in the same run):
      - rollout/ep_rew_mean : rolling mean reward over the last 100 episodes
      - train/loss          : TD-error loss
      - train/learning_rate : current learning rate
    """

    def __init__(self) -> None:
        super().__init__()
        self._writer: SummaryWriter | None = None
        self._episode_reward = 0.0

    def _on_training_start(self) -> None:
        from stable_baselines3.common.logger import TensorBoardOutputFormat
        for fmt in self.model._logger.output_formats:
            if isinstance(fmt, TensorBoardOutputFormat):
                self._writer = fmt.writer
                return
        self._writer = None

    def _on_step(self) -> bool:
        if self._writer is None:
            return True

        self._episode_reward += float(self.locals["rewards"][0])

        self._writer.add_scalar("training/epsilon",
                                self.model.exploration_rate,
                                self.num_timesteps)

        if self.locals["dones"][0]:
            self._writer.add_scalar("training/episode_reward",
                                    self._episode_reward,
                                    self.num_timesteps)
            self._episode_reward = 0.0

        return True


# ── Environment Builders ───────────────────────────────────────────────────────

def build_training_environment(seed: int) -> VecFrameStack:
    """Create a vectorised, preprocessed MsPacman environment for training.

    Applies the standard Atari preprocessing pipeline via make_atari_env +
    VecFrameStack:
      - Grayscale conversion and 84×84 resize
      - Frame skipping (repeat each action 4 steps)
      - Terminal-on-life-loss (3 lives → 3 mini-episodes per game)
      - Frame stacking (last N_STACK frames as one observation)

    Note on MsPacman lives: the game starts with 3 lives. terminal_on_life_loss
    is set by AtariWrapper inside make_atari_env, which treats each life as a
    separate training episode.  This increases data efficiency early in training.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        A VecFrameStack-wrapped vectorised environment ready for DQN.
    """
    env = make_atari_env(ENV_ID, n_envs=1, seed=seed)
    env = VecFrameStack(env, n_stack=N_STACK)
    return env


def build_playing_environment() -> VecFrameStack:
    """Create a human-rendered MsPacman environment for watching the agent play.

    Differences from the training environment:
      - render_mode="human" opens a visible game window.
      - clip_reward=False: show the real score (not clipped {-1, 0, +1}).

    Returns:
        A VecFrameStack-wrapped vectorised environment with a human-visible window.
    """
    def _make_single_env() -> AtariWrapper:
        base_env = gym.make(ENV_ID, render_mode="human")
        return AtariWrapper(base_env, terminal_on_life_loss=True, clip_reward=False)

    env = DummyVecEnv([_make_single_env])
    env = VecFrameStack(env, n_stack=N_STACK)
    return env


# ── Core Logic ─────────────────────────────────────────────────────────────────

def train_agent(
    model_path: str,
    timesteps: int,
    seed: int,
    tensorboard_log: str,
    hparams: dict | None = None,
) -> float:
    """Train a DQN agent on MsPacman-v5 and save the model.

    When called without `hparams` the function uses the MsPacman-tuned defaults
    described below.

    Hyperparameter rationale for MsPacman (300k-step budget):

      learning_rate        1e-4   — same as Breakout; fast enough without
                                    overshooting in 300k steps
      buffer_size        100_000  — larger than Breakout (50k) because the maze
                                    generates far more diverse (s, a, r, s')
                                    tuples; a richer buffer reduces correlation
      learning_starts     20_000  — 20% of budget; ensures the buffer contains
                                    varied maze positions before learning begins
      batch_size              64  — stable gradient estimates
      gamma                 0.99  — high discount: eating a ghost requires
                                    planning several steps ahead (collect pellet,
                                    then chase)
      train_freq               4  — one gradient update every 4 env steps
      target_update_interval 2_000 — ~148 syncs over 300k; less frequent than
                                    Breakout (1k) to dampen the noisy multi-
                                    objective reward signal
      exploration_fraction  0.20  — ε decays over 60k steps; the larger maze
                                    requires more random exploration than Breakout
                                    to discover all pellet clusters
      exploration_final_eps 0.01  — standard 1% random floor

    Args:
        model_path:      Path (without .zip) where the trained model is saved.
        timesteps:       Total environment steps to train for.
        seed:            Random seed for reproducibility.
        tensorboard_log: Directory where TensorBoard event files are written.
        hparams:         Optional hyperparameter dict from sweep runner.

    Returns:
        Mean episode reward over the last episodes in SB3's ep_info_buffer.
    """
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    if hparams is None:
        hparams = dict(
            env_id=ENV_ID,
            learning_rate=1e-4,
            buffer_size=100_000,
            learning_starts=20_000,
            batch_size=64,
            gamma=0.99,
            train_freq=4,
            target_update_interval=2_000,
            exploration_fraction=0.20,
            exploration_final_eps=0.01,
            timesteps=timesteps,
            seed=seed,
        )

    _tb_writer = SummaryWriter(log_dir=tensorboard_log)
    _tb_writer.add_hparams(hparams, metric_dict={"hparam/episode_reward": 0})
    _tb_writer.close()

    env = build_training_environment(seed=seed)

    model = DQN(
        policy="CnnPolicy",
        env=env,
        learning_rate=hparams["learning_rate"],
        buffer_size=hparams["buffer_size"],
        learning_starts=hparams["learning_starts"],
        batch_size=hparams["batch_size"],
        tau=1.0,
        gamma=hparams["gamma"],
        train_freq=hparams["train_freq"],
        gradient_steps=1,
        target_update_interval=hparams["target_update_interval"],
        exploration_fraction=hparams["exploration_fraction"],
        exploration_final_eps=hparams["exploration_final_eps"],
        verbose=1,
        tensorboard_log=tensorboard_log,
        seed=seed,
    )

    model.learn(
        total_timesteps=timesteps,
        callback=TensorBoardCallback(),
        progress_bar=True,
    )
    model.save(model_path)
    env.close()
    print(f"Model saved → {model_path}.zip")

    if model.ep_info_buffer:
        return float(np.mean([ep["r"] for ep in model.ep_info_buffer]))
    return 0.0


def play_agent(model_path: str, episodes: int) -> None:
    """Load a trained model and watch it play in a visible game window.

    Each full game (all 3 lives exhausted) counts as one episode.

    Args:
        model_path: Path to the saved model (with or without .zip extension).
        episodes:   Number of full games to play before exiting.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not os.path.exists(f"{model_path}.zip"):
        raise FileNotFoundError(
            f"Model not found: {model_path}.zip\n"
            "Run with --mode train first to create a model."
        )

    env = build_playing_environment()
    model = DQN.load(model_path, env=env)

    completed = 0
    obs = env.reset()
    episode_reward = 0.0

    while completed < episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        episode_reward += float(rewards[0])

        if dones[0]:
            # lives > 0 means a mid-game life loss (auto-reset continues).
            # lives == 0 means all 3 lives exhausted → count as one episode.
            if infos[0].get("lives", 0) == 0:
                completed += 1
                print(f"Episode {completed}/{episodes}  reward: {episode_reward:.2f}")
                episode_reward = 0.0

    env.close()


# ── Sweep ──────────────────────────────────────────────────────────────────────

def run_sweep(
    sweep_path: str,
    default_timesteps: int,
    seed: int,
    base_log_dir: str,
    best_model_path: str,
) -> None:
    """Run all experiments from a JSON config and save the best model.

    Each experiment uses its own ``timesteps`` if defined; otherwise falls
    back to ``default_timesteps``.  TensorBoard logs land in
    ``<base_log_dir>/sweep/<experiment_name>/`` so all runs are comparable.

    Args:
        sweep_path:        Path to the JSON file with experiment configs.
        default_timesteps: Fallback budget for experiments without ``timesteps``.
        seed:              Random seed applied to every experiment.
        base_log_dir:      Root TensorBoard log directory.
        best_model_path:   Where to save the winning model (without .zip).
    """
    with open(sweep_path) as f:
        configs = json.load(f)

    tmp_model_dir = Path("models") / "_sweep_tmp"
    tmp_model_dir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[str, float]] = []
    total = len(configs)

    for idx, cfg in enumerate(configs, start=1):
        name = cfg.get("name", f"exp_{idx:02d}")
        note = cfg.get("note", "")
        exp_timesteps = cfg.get("timesteps", default_timesteps)
        print(f"Experiment {idx}/{total}: {name}  ({exp_timesteps:,} steps)")
        if note:
            print(f"  {note}")
        print(f"{'='*60}")

        hparams = {
            "env_id":                 ENV_ID,
            "learning_rate":          cfg["learning_rate"],
            "buffer_size":            cfg["buffer_size"],
            "learning_starts":        cfg["learning_starts"],
            "batch_size":             cfg["batch_size"],
            "gamma":                  cfg["gamma"],
            "train_freq":             cfg["train_freq"],
            "target_update_interval": cfg["target_update_interval"],
            "exploration_fraction":   cfg["exploration_fraction"],
            "exploration_final_eps":  cfg["exploration_final_eps"],
            "timesteps":              exp_timesteps,
            "seed":                   seed,
        }

        model_path = str(tmp_model_dir / name)
        log_dir   = f"{base_log_dir}/sweep/{name}"

        score = train_agent(
            model_path=model_path,
            timesteps=exp_timesteps,
            seed=seed,
            tensorboard_log=log_dir,
            hparams=hparams,
        )
        results.append((name, score))
        print(f"  → final mean reward: {score:.2f}")

    results.sort(key=lambda x: x[1], reverse=True)
    best_name, best_score = results[0]

    print(f"\n{'='*60}")
    print("Sweep complete — results ranked by final mean reward:")
    for rank, (name, score) in enumerate(results, start=1):
        marker = "  BEST" if rank == 1 else ""
        print(f"  {rank:2d}. {name:<35s}  {score:7.2f}{marker}")
    print(f"{'='*60}")

    Path(best_model_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(tmp_model_dir / f"{best_name}.zip"), f"{best_model_path}.zip")
    shutil.rmtree(tmp_model_dir)

    print(f"\nBest model ({best_name}, score={best_score:.2f}) saved → {best_model_path}.zip")
    print(f"TensorBoard logs: {base_log_dir}/sweep/")


def inspect_model(model_path: str) -> None:
    """Load a saved model and print its hyperparameters.

    Args:
        model_path: Path to the saved model (with or without .zip extension).
    """
    if not os.path.exists(f"{model_path}.zip"):
        raise FileNotFoundError(f"Model not found: {model_path}.zip")

    model = DQN.load(model_path)

    params = {
        "policy":                  model.policy_class.__name__,
        "learning_rate":           model.learning_rate,
        "buffer_size":             model.buffer_size,
        "learning_starts":         model.learning_starts,
        "batch_size":              model.batch_size,
        "tau":                     model.tau,
        "gamma":                   model.gamma,
        "train_freq":              model.train_freq,
        "gradient_steps":          model.gradient_steps,
        "target_update_interval":  model.target_update_interval,
        "exploration_fraction":    model.exploration_fraction,
        "exploration_final_eps":   model.exploration_final_eps,
        "num_timesteps_trained":   model.num_timesteps,
    }

    print(f"\n── Saved model: {model_path}.zip")
    for key, value in params.items():
        print(f"  {key:30s}: {value}")
    print("─" * 55 + "\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or watch a DQN agent on ALE/MsPacman-v5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["train", "play", "inspect", "sweep"], required=True,
        help="'train' single run, 'play' watch agent, 'inspect' print params, "
             "'sweep' run all experiments from --sweep-file.",
    )
    parser.add_argument(
        "--sweep-file", default="sweep_configs_mspacman.json",
        help="Path to JSON file with experiment configs (used by --mode sweep and --experiment).",
    )
    parser.add_argument(
        "--experiment", default=None,
        help="Name of a single experiment in --sweep-file to run with --mode train.",
    )
    parser.add_argument(
        "--model-path", default="models/mspacman_dqn",
        help="Path to save (train) or load (play/inspect) the model (without .zip).",
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Total training steps. Overrides the JSON value when set. "
             "Defaults to the JSON value, or 300k if neither is specified.",
    )
    parser.add_argument(
        "--episodes", type=int, default=3,
        help="Number of full games to play in play mode.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tensorboard-log", default="logs/mspacman_dqn",
        help="Directory for TensorBoard logs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "train":
        hparams = None
        timesteps = args.timesteps or 300_000

        if args.experiment:
            with open(args.sweep_file) as f:
                configs = {c["name"]: c for c in json.load(f)}
            if args.experiment not in configs:
                raise ValueError(
                    f"Experiment '{args.experiment}' not found in {args.sweep_file}.\n"
                    f"Available: {', '.join(configs)}"
                )
            cfg = configs[args.experiment]
            timesteps = args.timesteps or cfg.get("timesteps", 300_000)
            hparams = {
                "env_id":                 ENV_ID,
                "learning_rate":          cfg["learning_rate"],
                "buffer_size":            cfg["buffer_size"],
                "learning_starts":        cfg["learning_starts"],
                "batch_size":             cfg["batch_size"],
                "gamma":                  cfg["gamma"],
                "train_freq":             cfg["train_freq"],
                "target_update_interval": cfg["target_update_interval"],
                "exploration_fraction":   cfg["exploration_fraction"],
                "exploration_final_eps":  cfg["exploration_final_eps"],
                "timesteps":              timesteps,
                "seed":                   args.seed,
            }
            print(f"Loaded experiment '{args.experiment}' from {args.sweep_file}")

        train_agent(
            model_path=args.model_path,
            timesteps=timesteps,
            seed=args.seed,
            tensorboard_log=args.tensorboard_log,
            hparams=hparams,
        )

    elif args.mode == "play":
        play_agent(model_path=args.model_path, episodes=args.episodes)

    elif args.mode == "sweep":
        run_sweep(
            sweep_path=args.sweep_file,
            default_timesteps=args.timesteps or 300_000,
            seed=args.seed,
            base_log_dir=args.tensorboard_log,
            best_model_path=args.model_path,
        )

    else:
        inspect_model(model_path=args.model_path)


if __name__ == "__main__":
    main()
