"""Regression tests for the corrected engine semantics (2026-09-07 audit).

Covers the four Phase-1 engine fixes from RECOVERY_PLAN.md:

1. Lawnmowers exist (created on reset, rolled in within one step) and a lane's
   first leak triggers the mower instead of losing the episode.
2. Upgrade planting consumes its base plant (Twin Sunflower eats Sunflower);
   the base plant no longer survives beneath the upgrade.
3. Cooldown observations encode 0.0 = ready, rising monotonically toward 1.0
   while recharging, back to 0.0 when ready.
4. reset() drains the widget safe-delete list, so repeated resets do not leak
   one full board of entity pools per episode.
5. Endless stage chaining (Phase 6): the stage boundary advances the live
   board in place (plants/sun/seed bank persist, wave machine resets with
   stage-scaled difficulty) instead of terminating the episode.

These tests need the rebuilt pvz_env extension and boot the real engine
(slow). They skip when the module cannot be imported.
"""

import ctypes
import os
import sys
import unittest

import torch  # noqa: F401  -- brings up DLL dirs the bridge links against

_BUILD_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "pvz-portable", "build")
if os.path.isdir(_BUILD_DIR):
    sys.path.insert(0, _BUILD_DIR)

try:
    if os.name == "nt" and os.path.isdir(r"D:\msys2\ucrt64\bin"):
        os.add_dll_directory(r"D:\msys2\ucrt64\bin")
    pvz_env = __import__("pvz_env")
except Exception:  # pragma: no cover - module or toolchain missing
    pvz_env = None


WAIT = 0
PLANT_OFF = 46  # 1 (wait) + 45 (shovel cells)
# Deck slot indices 0-9 in the bridge seed bank.
SLOT_SUNFLOWER = 0
SLOT_TWIN = 1
# SeedType enum + 1 as written into spatial channel 0.
SPATIAL_SUNFLOWER = 2.0
SPATIAL_TWIN = 42.0


def plant_action(seed: int, row: int, col: int) -> int:
    return PLANT_OFF + seed * 45 + row * 9 + col


def shovel_action(row: int, col: int) -> int:
    return 1 + row * 9 + col


@unittest.skipUnless(pvz_env, "pvz_env module unavailable")
class EngineSemanticsTest(unittest.TestCase):
    """Semantic checks against the real engine (one engine per process)."""

    @classmethod
    def setUpClass(cls):
        assert pvz_env is not None  # guaranteed by skipUnless
        cls.env = pvz_env.PvZEnv()

    def obs(self):
        return self.env.get_obs()

    # ── Fix 1: lawnmowers ──────────────────────────────────────────────

    def test_mowers_roll_in_within_first_step(self):
        obs = self.obs()  # fresh reset below leaves the engine at t=0
        self.env.reset(1, 283001)
        obs = self.obs()
        at_reset = [float(v) for v in obs["global"][12:17]]
        self.assertEqual(at_reset, [0.0] * 5, "mowers should still be rolling in")

        obs, _, _, _, _, _ = self.env.step(WAIT)
        after_step = [float(v) for v in obs["global"][12:17]]
        self.assertEqual(
            after_step, [1.0] * 5, "all five mowers must be READY after 100 frames"
        )

    def test_first_lane_leak_triggers_mower_and_episode_continues(self):
        self.env.reset(1, 283002)
        triggered = 0
        for _ in range(220):
            obs, mask, _, done, truncated, info = self.env.step(WAIT)
            if info["triggered_mowers"] > triggered:
                triggered = info["triggered_mowers"]
                ready = [float(v) for v in obs["global"][12:17]]
                self.assertFalse(
                    bool(done),
                    "first leak must be saved by the mower, not end the episode",
                )
                self.assertFalse(bool(truncated))
                self.assertEqual(
                    ready.count(1.0), 4, "exactly one mower should be consumed"
                )
                # Episode must still be running several steps after the save.
                for _ in range(3):
                    obs, _, _, done, _, info = self.env.step(WAIT)
                    self.assertFalse(bool(done), "episode continues after mower save")
                self.assertGreaterEqual(info["triggered_mowers"], 1)
                return
        self.fail(f"no mower ever triggered in 220 wait steps (triggered={triggered})")

    # ── Fix 2: upgrade consumes base plant ─────────────────────────────

    def test_twin_upgrade_consumes_sunflower(self):
        self.env.reset(1, 283003)
        self.env.set_sun_money(3000)

        obs, mask, _, _, _, _ = self.env.step(plant_action(SLOT_SUNFLOWER, 0, 0))
        self.assertEqual(
            float(obs["spatial"][0][0][0]), SPATIAL_SUNFLOWER, "sunflower planted"
        )

        obs, mask, _, _, _, _ = self.env.step(plant_action(SLOT_TWIN, 0, 0))
        self.assertEqual(
            float(obs["spatial"][0][0][0]),
            SPATIAL_TWIN,
            "cell must show only the Twin; the Sunflower base must be consumed",
        )

        # One shovel must empty the cell completely (the old bug left the
        # surviving base plant behind, needing a second shovel).
        obs, mask, _, _, _, _ = self.env.step(shovel_action(0, 0))
        self.assertEqual(float(obs["spatial"][0][0][0]), 0.0, "cell fully emptied")

    # ── Fix 3: cooldown observation encoding ───────────────────────────

    def test_cooldown_obs_zero_when_ready_rises_while_recharging(self):
        self.env.reset(1, 283004)
        obs = self.obs()
        for slot in range(10):
            self.assertEqual(
                float(obs["global"][2 + slot]), 0.0, f"slot {slot} ready at reset"
            )

        obs, mask, _, _, _, _ = self.env.step(plant_action(SLOT_SUNFLOWER, 1, 0))
        v1 = float(obs["global"][2 + SLOT_SUNFLOWER])
        self.assertGreater(v1, 0.0, "recharge in progress reads > 0")
        self.assertLess(v1, 1.0)
        # Sun cooldown masks its plant actions while recharging.
        self.assertFalse(bool(mask[plant_action(SLOT_SUNFLOWER, 2, 2)]))
        # Other slots remain ready.
        self.assertEqual(float(obs["global"][2 + SLOT_TWIN]), 0.0)

        prev = v1
        saw_increase = False
        ready_again = False
        for _ in range(20):
            obs, mask, _, _, _, _ = self.env.step(WAIT)
            v = float(obs["global"][2 + SLOT_SUNFLOWER])
            if v > 0.0:
                if v > prev:
                    saw_increase = True
                prev = v
            else:
                ready_again = True
                self.assertTrue(bool(mask[plant_action(SLOT_SUNFLOWER, 2, 2)]))
                break
        self.assertTrue(saw_increase, "cooldown value must rise while recharging")
        self.assertTrue(ready_again, "cooldown must return to exactly 0.0")

    # ── Fix 4: no board leak per reset ─────────────────────────────────

    def test_repeated_resets_do_not_leak_boards(self):
        class PMC(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_uint32),
                ("PageFaultCount", ctypes.c_uint32),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        def rss_mb():
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            k32 = ctypes.windll.kernel32
            # The pseudo-handle is 0xFFFF...FF; the default c_int restype
            # truncates it and the call fails with ERROR_INVALID_HANDLE.
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            k32.K32GetProcessMemoryInfo.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(PMC),
                ctypes.c_uint32,
            ]
            ok = k32.K32GetProcessMemoryInfo(
                k32.GetCurrentProcess(),
                ctypes.byref(pmc),
                ctypes.sizeof(PMC),
            )
            if not ok:
                self.skipTest("K32GetProcessMemoryInfo unavailable")
            return pmc.WorkingSetSize / (1024 * 1024)

        if os.name != "nt":
            self.skipTest("Windows-only RSS probe")

        for i in range(3):  # warm-up allocations
            self.env.reset(1, 283100 + i)
        baseline = rss_mb()
        for i in range(25):
            self.env.reset(1, 283200 + i)
        growth = rss_mb() - baseline
        # Each leaked board holds six full entity pools; 25 boards would add
        # far more than this bound. Generous enough to absorb allocator noise.
        self.assertLess(growth, 150.0, f"RSS grew {growth:.1f} MB over 25 resets")


# ── Fix 5 / Phase 6: endless stage chaining ─────────────────────────


class StageChainingTest(unittest.TestCase):
    """The stage boundary advances the live board in place instead of ending."""

    @classmethod
    def setUpClass(cls):
        assert pvz_env is not None  # guaranteed by skipUnless on the module
        cls.env = pvz_env.PvZEnv()

    def test_chain_advances_stage_and_persists_plants(self):
        self.env.reset(1, 283010)
        self.env.set_chain_stages(True)

        # A Sunflower planted before the boundary must survive the chain.
        obs, _, _, _, _, _ = self.env.step(plant_action(SLOT_SUNFLOWER, 0, 0))
        self.assertEqual(float(obs["spatial"][0][0][0]), SPATIAL_SUNFLOWER)

        self.env.debug_trigger_stage_end()
        obs, mask, reward, done, truncated, info = self.env.step(WAIT)
        self.assertTrue(
            bool(info["stage_complete"]), "boundary step reports completion"
        )
        self.assertFalse(bool(done), "chaining must keep the episode alive")
        self.assertFalse(bool(truncated))
        self.assertEqual(int(info["stage"]), 1, "stage counter advanced")
        self.assertEqual(
            int(info["wave"]), 21, "stage-2 wave 1 reads as absolute wave 21"
        )
        self.assertEqual(
            int(info["num_waves"]), 20, "wave machine reset for the new stage"
        )
        self.assertEqual(
            float(obs["spatial"][0][0][0]),
            SPATIAL_SUNFLOWER,
            "plants persist across the stage boundary",
        )

        # The chained stage's wave machine must actually spawn zombies.
        spawned = False
        for _ in range(80):
            obs, _, _, done, _, info = self.env.step(WAIT)
            if float(info["zombie_min_x"]) < 700.0:
                spawned = True
                break
            self.assertFalse(bool(done), "episode must stay alive on the chained stage")
        self.assertTrue(spawned, "no zombie spawned on the chained stage in 80 steps")

    def test_chain_off_still_terminates_at_stage_end(self):
        self.env.reset(1, 283011)
        self.env.set_chain_stages(False)
        self.env.debug_trigger_stage_end()
        obs, _, _, done, _, info = self.env.step(WAIT)
        self.assertTrue(bool(info["stage_complete"]))
        self.assertTrue(bool(done), "default mode must terminate at stage end")


class CoffeeWakeTest(unittest.TestCase):
    """Night-shrooms sleep on the day lawn; Coffee Bean wakes them."""

    DEFAULT_DECK = [1, 41, 39, 44, 42, 10, 30, 36, 17, 20]
    COFFEE_DECK = [1, 41, 39, 44, 42, 10, 30, 35, 17, 20]

    def setUp(self):
        assert pvz_env is not None  # guaranteed by skipUnless on the module
        self.env = pvz_env.PvZEnv()

    def tearDown(self):
        # Restore the default deck so later classes' resets are unaffected.
        self.env.set_deck(self.DEFAULT_DECK)

    def test_coffee_wakes_sleeping_fume(self):
        self.env.set_deck(self.COFFEE_DECK)
        self.env.reset(1, 277782)
        self.env.set_sun_money(9990)
        self.env.step(plant_action(5, 0, 0))  # Fume-shroom at (0,0)
        plants = self.env.debug_sleep_state()["plants"]
        fumes = [p for p in plants if p["seed"] == 10]
        self.assertTrue(fumes, "Fume-shroom planted")
        self.assertTrue(fumes[0]["asleep"], "night-shroom plants asleep on day lawn")

        mask = self.env.get_action_mask()
        self.assertTrue(bool(mask[plant_action(7, 0, 0)]), "coffee legal on a sleeper")

        self.env.step(plant_action(7, 0, 0))  # Coffee Bean on the Fume
        self.env.step(WAIT)
        plants = self.env.debug_sleep_state()["plants"]
        fumes = [p for p in plants if p["seed"] == 10]
        self.assertTrue(
            fumes and not fumes[0]["asleep"], "coffee woke the Fume-shroom"
        )


if __name__ == "__main__":
    unittest.main()
