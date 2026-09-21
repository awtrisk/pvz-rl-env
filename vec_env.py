"""Minimal vectorized PvZ environment using one process per env.

The C++ engine is a singleton per process (global gLawnApp), so each worker
must live in its own process. We use multiprocessing.Pipe and spawn context.
"""

import multiprocessing as mp
import os

import numpy as np

from pvz_portable_gym import PvZGymEnv


def _log(msg):
    print(f"[PVZ_VEC_ENV] {msg}", flush=True)


def _worker(conn, env_kwargs, reset_kwargs):
    """Worker process that owns one PvZGymEnv."""
    try:
        _log("worker starting")
        env_kwargs = dict(env_kwargs)
        savedir = env_kwargs.get("savedir")
        if savedir:
            # Each native LawnApp writes profile state during shutdown. Sharing
            # one save directory lets 20 workers corrupt each other's files.
            savedir = os.path.join(savedir, f"worker_{os.getpid()}")
            os.makedirs(savedir, exist_ok=True)
            env_kwargs["savedir"] = savedir
        env = PvZGymEnv(**env_kwargs)
        _log("worker env created")
    except Exception as e:
        _log(f"worker env creation failed: {e}")
        conn.send(e)
        conn.close()
        return

    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "step":
                conn.send(env.step(data))
            elif cmd == "reset":
                conn.send(env.reset(options=data))
            elif cmd == "set_options":
                conn.send(None)
            elif cmd == "set_sun_penalty_scale":
                env.set_sun_penalty_scale(data)
                conn.send(None)
            elif cmd == "set_plant_bonus_scale":
                env.set_plant_bonus_scale(data)
                conn.send(None)
            elif cmd == "set_cooldown_scale":
                env.set_cooldown_scale(data)
                conn.send(None)
            elif cmd == "set_chain_stages":
                env.set_chain_stages(data)
                conn.send(None)
            elif cmd == "set_deck":
                env.set_deck(data)
                conn.send(None)
            elif cmd == "close":
                conn.send(None)
                break
            elif cmd == "get_spaces":
                conn.send((env.observation_space, env.action_space))
    except Exception as e:
        _log(f"worker loop error: {e}")
        raise
    finally:
        env.close()
        conn.close()


class PvZVecEnv:
    """Synchronous vector env for PvZ.  Each env runs in its own process."""

    def __init__(
        self,
        num_envs: int,
        resdir: str,
        savedir: str,
        start_wave: int = 1,
        seed: int | None = None,
    ):
        self.num_envs = num_envs
        self.start_wave = start_wave
        self.env_kwargs = {"resdir": resdir, "savedir": savedir, "chdir": True}
        self.reset_kwargs = {"wave": start_wave}
        self.reset_kwargs_per_env = None
        self.seed = seed
        self.episode_counts = np.zeros(num_envs, dtype=np.int64)

        self._conns = []
        self._procs = []
        ctx = mp.get_context("spawn")
        for _ in range(num_envs):
            parent_conn, child_conn = ctx.Pipe()
            p = ctx.Process(
                target=_worker, args=(child_conn, self.env_kwargs, self.reset_kwargs)
            )
            p.start()
            self._conns.append(parent_conn)
            self._procs.append(p)
            parent_conn.send(("get_spaces", None))

        self.observation_space, self.action_space = self._conns[0].recv()
        # Drain remaining get_spaces responses from other workers.
        for conn in self._conns[1:]:
            conn.recv()

    def _reset_options(self, index):
        base = (self.reset_kwargs_per_env or [self.reset_kwargs] * self.num_envs)[index]
        options = dict(base)
        if self.seed is not None:
            options["seed"] = int(
                self.seed + index + self.episode_counts[index] * self.num_envs
            )  # pi-lens-ignore: unchecked-throwing-call-python
        self.episode_counts[index] += 1
        return options

    def reset(self):
        options = [self._reset_options(i) for i in range(self.num_envs)]
        for conn, reset_kwargs in zip(self._conns, options, strict=True):
            conn.send(("reset", reset_kwargs))
        obs, infos = zip(*[conn.recv() for conn in self._conns], strict=True)
        return self._stack_obs(obs), list(infos)

    def step(self, actions):
        for conn, action in zip(self._conns, actions, strict=True):
            conn.send(
                ("step", int(action))
            )  # pi-lens-ignore: unchecked-throwing-call-python
        results = [conn.recv() for conn in self._conns]
        obs, rewards, dones, truncated, infos = zip(*results, strict=True)
        return (
            self._stack_obs(obs),
            np.array(rewards),
            np.array(dones),
            np.array(truncated),
            list(infos),
        )

    def set_start_wave(self, wave):
        """Broadcast new reset options to all workers."""
        self.reset_kwargs["wave"] = wave
        for conn in self._conns:
            conn.send(("set_options", self.reset_kwargs))
        for conn in self._conns:
            conn.recv()

    def set_start_sun(self, sun):
        """Broadcast new starting sun to all workers via reset options."""
        self.reset_kwargs["sun"] = sun
        self.reset_kwargs_per_env = None
        for conn in self._conns:
            conn.send(("set_options", self.reset_kwargs))
        for conn in self._conns:
            conn.recv()

    def set_start_suns(self, suns):
        """Set one starting-sun value per worker for mixed-domain rollouts."""
        suns = list(suns)
        if len(suns) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} sun values, got {len(suns)}")
        self.reset_kwargs_per_env = [
            {**self.reset_kwargs, "sun": int(sun)} for sun in suns
        ]  # pi-lens-ignore: unchecked-throwing-call-python
        for conn, options in zip(self._conns, self.reset_kwargs_per_env, strict=True):
            conn.send(("set_options", options))
        for conn in self._conns:
            conn.recv()

    def set_sun_penalty_scale(self, scale):
        """Broadcast sun-penalty scale to all workers."""
        for conn in self._conns:
            conn.send(("set_sun_penalty_scale", scale))
        for conn in self._conns:
            conn.recv()

    def set_plant_bonus_scale(self, scale):
        """Broadcast new plant-bonus scale to all workers."""
        for conn in self._conns:
            conn.send(("set_plant_bonus_scale", scale))
        for conn in self._conns:
            conn.recv()

    def set_cooldown_scale(self, scale):
        """Broadcast a seed-cooldown multiplier to all workers."""
        for conn in self._conns:
            conn.send(("set_cooldown_scale", scale))
        for conn in self._conns:
            conn.recv()

    def set_cooldown_scales(self, scales):
        """Set one cooldown multiplier per worker for mixed-domain rollouts."""
        scales = list(scales)
        if len(scales) != self.num_envs:
            raise ValueError(
                f"expected {self.num_envs} cooldown scales, got {len(scales)}"
            )
        for conn, scale in zip(self._conns, scales, strict=True):
            conn.send(
                ("set_cooldown_scale", float(scale))
            )  # pi-lens-ignore: unchecked-throwing-call-python
        for conn in self._conns:
            conn.recv()

    def set_chain_stages(self, chain):
        """Endless mode: chain stage boundaries instead of terminating."""
        for conn in self._conns:
            conn.send(("set_chain_stages", bool(chain)))
        for conn in self._conns:
            conn.recv()

    def set_deck(self, deck):
        """Configure the 10-seed deck on every worker (applied next reset)."""
        for conn in self._conns:
            conn.send(("set_deck", [int(seed) for seed in deck]))  # pi-lens-ignore: unchecked-throwing-call-python
        for conn in self._conns:
            conn.recv()

    def reset_at(self, indices):
        """Reset specific envs and return their (obs, info) tuples."""
        indices = list(indices)
        for i in indices:
            self._conns[i].send(("reset", self._reset_options(i)))
        return [self._conns[i].recv() for i in indices]

    def close(self):
        for conn in self._conns:
            conn.send(("close", None))
        for p in self._procs:
            p.join()

    @staticmethod
    def _stack_obs(obs):
        return {
            "spatial": np.stack([o["spatial"] for o in obs]),
            "global": np.stack([o["global"] for o in obs]),
        }
