"""Minimal PvZGymEnv demo: obs v2, sparse reward, custom deck, endless mode.

Run from the repo root:  PYTHONPATH=. python examples/quickstart.py
A random policy survives a few waves; swap the action choice for your agent.
"""

import numpy as np

import pvz_portable_gym

# Seed types: Sunflower, Twin Sunflower, Melon-pult, Winter Melon,
# Gloom-shroom, Fume-shroom, Pumpkin, Garlic, Squash, Jalapeno.
DECK = [1, 41, 39, 44, 42, 10, 30, 36, 17, 20]


def main() -> None:
    env = pvz_portable_gym.PvZGymEnv(
        reward_mode="sparse",
        obs_version=2,
        deck=DECK,
    )
    env.set_chain_stages(True)  # endless: survive past every stage boundary

    obs, info = env.reset(seed=0)
    print("spatial", obs["spatial"].shape, "global", obs["global"].shape)

    steps = 0
    done = truncated = False
    while not (done or truncated):
        legal = np.flatnonzero(info["action_mask"])
        action = int(env.np_random.choice(legal))  # pi-lens-ignore: unchecked-throwing-call-python
        obs, reward, done, truncated, info = env.step(action)
        steps += 1

    print(
        f"episode over after {steps} steps: "
        f"absolute wave {info['wave']}, stage {info['stage']}, "
        f"lost={info['lost']}, stage_complete={info['stage_complete']}"
    )


if __name__ == "__main__":
    main()
