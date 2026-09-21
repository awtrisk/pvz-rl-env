"""Record a clean native-frame PvZ showcase.

The video deliberately keeps the game as the hero: no dashboard, no fake grid,
no UI pasted over the lawn.  It adds only quiet letterbox captions around the
real SDL/OpenGL framebuffer.

Two-stage montage example::

    python scripts/record.py \
        --checkpoint showcase/plan_wave20_best.pt \
        --output showcase/endless_montage.mp4 \
        --waves 1,21 --seeds 272000,272003
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from curriculum import CurriculumManager
from network import PvZActorCritic
from pvz_portable_gym import PvZGymEnv


BG = (10, 13, 19)
WHITE = (240, 241, 236)
MUTED = (139, 148, 155)
ACCENT = (238, 184, 72)
GREEN = (111, 211, 143)
RED = (230, 94, 91)


def _font(size: int, bold: bool = False):
    names = (
        r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
        r"/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else r"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for name in names:
        if os.path.exists(name):
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


F11 = _font(11)
F12 = _font(12)
F13B = _font(13, True)
F20B = _font(20, True)
F38B = _font(38, True)


def _parse_ints(value: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer list: {value!r}") from exc
    if not values:
        raise argparse.ArgumentTypeError("the list cannot be empty")
    return values


def _action_label(action: int) -> str:
    if action == 0:
        return "WAIT"
    if action < 46:
        cell = action - 1
        return f"SHOVEL  R{cell // 9 + 1} C{cell % 9 + 1}"
    seed, cell = divmod(action - 46, 45)
    row, col = divmod(cell, 9)
    names = ("SUN", "TWIN", "MELON", "WINTER", "GLOOM", "FUME", "PUMP", "GARLIC", "SQUASH", "JALA")
    return f"{names[seed]}  R{row + 1} C{col + 1}"


def _frame(native_frame, info, *, action: int, seed: int, stage: int, start_sun: int, terminal: str | None = None):
    """Put the native 4:3 game in a quiet 16:9 presentation frame."""
    native = Image.fromarray(np.asarray(native_frame, dtype=np.uint8), mode="RGB")
    canvas = Image.new("RGB", (1280, 720), BG)
    draw = ImageDraw.Draw(canvas)

    # 800x600 -> 960x720. The 160px gutters carry only captions.
    canvas.paste(native.resize((960, 720), Image.Resampling.LANCZOS), (160, 0))

    wave = int(info.get("wave", 1))
    state = terminal or "LIVE"
    state_color = GREEN if state in {"LIVE", "STAGE CLEAR"} else RED

    # Left gutter: restrained title and one action caption.
    draw.line((42, 42, 42, 678), fill=(45, 50, 56), width=1)
    draw.text((58, 42), "PVZ", fill=WHITE, font=F20B)
    draw.text((59, 70), "RL RUN", fill=ACCENT, font=F11)
    draw.text((58, 603), "POLICY", fill=MUTED, font=F11)
    draw.text((58, 622), _action_label(action), fill=WHITE, font=F12)
    draw.text((58, 638), f"START SUN {start_sun}", fill=MUTED, font=F11)
    draw.text((58, 657), f"SEED {seed}", fill=MUTED, font=F11)

    # Right gutter: one large counter, not a metrics panel.
    draw.text((1150, 42), f"{wave:02d}", fill=WHITE, font=F38B, anchor="ra")
    draw.text((1150, 91), "WAVE", fill=ACCENT, font=F13B, anchor="ra")
    draw.line((1135, 120, 1235, 120), fill=(60, 66, 72), width=1)
    draw.text((1150, 139), f"STAGE {stage:02d}", fill=WHITE, font=F13B, anchor="ra")
    draw.text((1150, 164), state, fill=state_color, font=F11, anchor="ra")

    if terminal:
        # Only the result gets a centered title; the running footage stays clean.
        draw.rectangle((160, 0, 1120, 58), fill=(8, 10, 14))
        draw.text((640, 29), terminal, fill=state_color, font=F20B, anchor="mm")

    return np.asarray(canvas, dtype=np.uint8)


def _title_card(stage_wave: int, segment: int, seed: int, fps: int, seconds: float, start_sun: int = 0):
    canvas = Image.new("RGB", (1280, 720), BG)
    draw = ImageDraw.Draw(canvas)
    stage = (stage_wave - 1) // 20 + 1
    end_wave = stage * 20
    draw.text((640, 275), "SURVIVAL ENDLESS", fill=WHITE, font=F20B, anchor="mm")
    draw.text((640, 340), f"STAGE {stage:02d}", fill=ACCENT, font=F38B, anchor="mm")
    draw.text((640, 395), f"WAVE {stage_wave}  →  {end_wave}", fill=WHITE, font=F13B, anchor="mm")
    draw.line((520, 430, 760, 430), fill=(70, 75, 80), width=1)
    draw.text((640, 458), f"RUN {segment:02d}  /  SEED {seed}  /  START SUN {start_sun}", fill=MUTED, font=F11, anchor="mm")
    frame = np.asarray(canvas, dtype=np.uint8)
    return [frame] * max(1, int(fps * seconds))


def parse_args():
    parser = argparse.ArgumentParser(description="Record a native PvZ RL showcase.")
    parser.add_argument("--checkpoint", required=True, help="Path to a PPO checkpoint (.pt).")
    parser.add_argument("--output", default="showcase/endless_montage.mp4")
    parser.add_argument("--poster", default=None, help="Poster PNG; defaults beside the MP4.")
    parser.add_argument("--wave", type=int, default=None, help="One starting wave (legacy shorthand).")
    parser.add_argument("--waves", type=_parse_ints, default=None, help="Comma-separated starting waves, e.g. 1,21.")
    parser.add_argument("--seed", type=int, default=None, help="One seed (legacy shorthand).")
    parser.add_argument("--seeds", type=_parse_ints, default=None, help="Comma-separated seeds matching --waves.")
    parser.add_argument("--start-sun", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frame-interval", type=int, default=20, help="Native game frames between captured frames.")
    parser.add_argument("--title-seconds", type=float, default=1.2)
    parser.add_argument("--hold-final", type=float, default=2.0)
    parser.add_argument("--phase0-deck", action="store_true")
    parser.add_argument("--resdir", default="pvz-portable/")
    parser.add_argument("--savedir", default="pvz-portable/savedata/")
    return parser.parse_args()


def _segments(args) -> list[tuple[int, int]]:
    waves = args.waves if args.waves is not None else [args.wave if args.wave is not None else 1]
    seeds = args.seeds if args.seeds is not None else [args.seed if args.seed is not None else 272000 + i for i in range(len(waves))]
    if len(waves) != len(seeds):
        raise ValueError("--waves and --seeds must contain the same number of entries")
    return list(zip(waves, seeds))


def _load_agent(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state = checkpoint.get("agent_state", checkpoint)
    agent = PvZActorCritic().to(device)
    agent.load_state_dict(state)
    agent.eval()
    return agent


def _record_segment(agent, args, start_wave, seed, writer, segment_index, curriculum_mask):
    device = next(agent.parameters()).device
    env = PvZGymEnv(render_mode="rgb_array", resdir=args.resdir, savedir=args.savedir)
    env.set_cooldown_scale(1.0)
    obs, info = env.reset(options={"wave": start_wave, "sun": args.start_sun, "seed": seed})
    caches = agent.init_caches(batch_size=1, device=device)
    total_reward = 0.0
    steps = 0
    frames = 0
    terminal = None
    last_canvas = None
    action_counts = {"wait": 0, "shovel": 0, "plant": 0}

    try:
        info = dict(info)
        info["wave"] = start_wave
        info["stage"] = (start_wave - 1) // 20
        info["sun"] = args.start_sun
        display_stage = (start_wave - 1) // 20 + 1
        opening = _frame(
            env.render(), info, action=0, seed=seed, stage=display_stage, start_sun=args.start_sun
        )
        for _ in range(max(1, args.fps // 3)):
            writer.append_data(opening)
            frames += 1
        last_canvas = opening

        while steps < args.max_steps:
            spatial = torch.from_numpy(obs["spatial"]).unsqueeze(0).to(device)
            global_vec = torch.from_numpy(obs["global"]).unsqueeze(0).to(device)
            action_mask = torch.from_numpy(info["action_mask"]).unsqueeze(0).to(device)
            if args.phase0_deck:
                action_mask = action_mask * curriculum_mask
            with torch.no_grad():
                action, _, _, _, caches = agent.forward_step(
                    spatial, global_vec, action_mask, caches, deterministic=True
                )
            action_id = int(action.item())
            if action_id == 0:
                action_counts["wait"] += 1
            elif action_id < 46:
                action_counts["shovel"] += 1
            else:
                action_counts["plant"] += 1

            obs, reward, done, truncated, info, native_frames = env.record_step(
                action_id, frame_interval=args.frame_interval
            )
            total_reward += float(reward)
            steps += 1
            if done or truncated:
                terminal = "STAGE CLEAR" if info.get("stage_complete") else "ZOMBIES WON" if info.get("lost") else "STOPPED"

            for native_frame in native_frames:
                canvas = _frame(
                    native_frame,
                    info,
                    action=action_id,
                    seed=seed,
                    stage=display_stage,
                    start_sun=args.start_sun,
                    terminal=terminal,
                )
                writer.append_data(canvas)
                last_canvas = canvas
                frames += 1
            if done or truncated:
                break
    finally:
        env.close()

    return {
        "segment": segment_index,
        "start_wave": start_wave,
        "seed": seed,
        "steps": steps,
        "frames": frames,
        "final_wave": int(info.get("wave", start_wave)),
        "terminal_reason": "stage_complete" if info.get("stage_complete") else "lost" if info.get("lost") else "truncated" if terminal == "STOPPED" else "max_steps",
        "total_reward": total_reward,
        "action_counts": action_counts,
        "last_canvas": last_canvas,
    }


def _record_montage(args, segments):
    """Record each native segment in a fresh process, then join the footage."""
    import imageio.v2 as imageio

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for multi-stage montages")
    output = Path(args.output).expanduser().resolve()
    poster = Path(args.poster).expanduser().resolve() if args.poster else output.with_suffix(".poster.png")
    summary_path = output.with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="pvz-showcase-") as temp_name:
        temp = Path(temp_name)
        part_summaries = []
        part_paths = []
        for index, (wave, seed) in enumerate(segments, 1):
            part = temp / f"part-{index:02d}.mp4"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--checkpoint", str(Path(args.checkpoint).expanduser().resolve()),
                "--output", str(part),
                "--wave", str(wave),
                "--seed", str(seed),
                "--start-sun", str(args.start_sun),
                "--max-steps", str(args.max_steps),
                "--fps", str(args.fps),
                "--frame-interval", str(args.frame_interval),
                "--title-seconds", "0",
                "--hold-final", "0",
                "--resdir", args.resdir,
                "--savedir", args.savedir,
            ]
            if args.device:
                command += ["--device", args.device]
            if args.phase0_deck:
                command.append("--phase0-deck")
            subprocess.run(command, check=True)
            part_summaries.append(json.loads(part.with_suffix(".json").read_text(encoding="utf-8")))
            part_paths.append(part)

        ordered_paths = [part_paths[0]]
        for index, (wave, seed) in enumerate(segments[1:], 2):
            handoff = temp / f"handoff-{index:02d}.mp4"
            handoff_writer = imageio.get_writer(
                handoff, fps=args.fps, codec="libx264", quality=8, macro_block_size=1
            )
            stage = (wave - 1) // 20 + 1
            for frame in _title_card(wave, stage, seed, args.fps, args.title_seconds, args.start_sun):
                handoff_writer.append_data(frame)
            handoff_writer.close()
            ordered_paths.extend((handoff, part_paths[index - 1]))

        concat = temp / "concat.txt"
        concat.write_text(
            "".join(f"file '{path.name}'\n" for path in ordered_paths),
            encoding="utf-8",
        )
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", concat.name, "-c", "copy", str(output)],
            cwd=temp,
            check=True,
        )
        last_poster = Path(part_summaries[-1]["poster"])
        shutil.copyfile(last_poster, poster)

    clean_segments = []
    for index, summary in enumerate(part_summaries, 1):
        segment = dict(summary["segments"][0])
        segment["segment"] = index
        clean_segments.append(segment)
    summary = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "output": str(output),
        "poster": str(poster),
        "fps": args.fps,
        "frame_interval": args.frame_interval,
        "start_sun": args.start_sun,
        "segments": clean_segments,
        "terminal_wave": clean_segments[-1]["final_wave"],
        "note": "Native Survival Endless stage recordings joined with an explicit stage handoff card. Each segment uses a fresh engine process because the headless bridge ends at stage boundaries.",
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def record_rollout(args):
    import imageio.v2 as imageio

    segments = _segments(args)
    if len(segments) > 1:
        return _record_montage(args, segments)

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    poster = Path(args.poster).expanduser().resolve() if args.poster else output.with_suffix(".poster.png")
    summary_path = output.with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Loading {checkpoint} on {device}")
    agent = _load_agent(checkpoint, device)
    curriculum_mask = torch.from_numpy(CurriculumManager().seed_mask(agent.num_actions)).to(device)
    segments = _segments(args)

    writer = imageio.get_writer(output, fps=args.fps, codec="libx264", quality=8, macro_block_size=1)
    results = []
    try:
        for index, (start_wave, seed) in enumerate(segments, 1):
            if index > 1:
                for title_frame in _title_card(
                    start_wave, index, seed, args.fps, args.title_seconds, args.start_sun
                ):
                    writer.append_data(title_frame)
            result = _record_segment(agent, args, start_wave, seed, writer, index, curriculum_mask)
            results.append(result)
            print(f"segment={index} start={start_wave} end={result['final_wave']} steps={result['steps']} reason={result['terminal_reason']}")
            if index == len(segments) and result["last_canvas"] is not None:
                for _ in range(max(0, int(args.hold_final * args.fps))):
                    writer.append_data(result["last_canvas"])
    finally:
        writer.close()

    poster_image = results[-1].get("last_canvas") if results else None
    if poster_image is not None:
        Image.fromarray(poster_image).save(poster)

    clean_results = [{key: value for key, value in result.items() if key != "last_canvas"} for result in results]
    summary = {
        "checkpoint": str(checkpoint),
        "output": str(output),
        "poster": str(poster),
        "fps": args.fps,
        "frame_interval": args.frame_interval,
        "start_sun": args.start_sun,
        "segments": clean_results,
        "terminal_wave": clean_results[-1]["final_wave"] if clean_results else None,
        "note": "Each segment is a native Survival Endless stage recording; title cards mark the explicit stage handoff.",
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    print(json.dumps(record_rollout(parse_args()), indent=2))


if __name__ == "__main__":
    main()
