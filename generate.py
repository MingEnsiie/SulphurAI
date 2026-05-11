#!/usr/bin/env python3
"""
Sulphur 2  —  text-to-video / image-to-video generation.

Usage:
  # text-to-video
  python generate.py --prompt "a cat running on a beach"

  # image-to-video
  python generate.py --prompt "camera slowly zooms in" --image input.jpg

  # use local text-encoder path (skip HF download)
  python generate.py --prompt "..." --text-encoder /path/to/gemma-3-12b

Full options: python generate.py --help
"""

import argparse
import sys
from pathlib import Path

import torch

from src.progress import ProgressReporter


def parse_args():
    p = argparse.ArgumentParser(description="Sulphur 2 text-to-video generation")
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--image", default=None, help="input image for i2v mode")
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--frames", type=int, default=65, help="number of frames (must be 8n+1)")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--guidance-scale", type=float, default=3.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="outputs/output.mp4")
    p.add_argument(
        "--text-encoder",
        default=None,
        help="Local path or HF repo for Gemma 3 12B text encoder. "
             "Auto-detects Assets/models/gemma-3-12b-it-bnb-4bit, "
             "otherwise downloads unsloth/gemma-3-12b-it-bnb-4bit.",
    )
    return p.parse_args()


def save_video(frames, path: str, fps: int = 24):
    import imageio
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames_np = [(f.cpu().numpy() * 255).astype("uint8") for f in frames]
    progress = ProgressReporter("Saving video", len(frames_np))
    with imageio.get_writer(str(path), fps=fps) as w:
        for idx, frame in enumerate(frames_np, start=1):
            w.append_data(frame)
            progress.update(idx)
    progress.finish()
    print(f"Saved → {path}")


def main():
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).parent))
    from src import load_pipeline

    pipe = load_pipeline(
        text_encoder_path=args.text_encoder,
    )

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    progress = ProgressReporter("Generating", args.steps)

    if args.image:
        from PIL import Image
        image = Image.open(args.image).convert("RGB")
        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image=image,
            width=args.width,
            height=args.height,
            num_frames=args.frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            callback_on_step_end=progress.step_callback,
            callback_on_step_end_tensor_inputs=[],
        )
    else:
        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            width=args.width,
            height=args.height,
            num_frames=args.frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            callback_on_step_end=progress.step_callback,
            callback_on_step_end_tensor_inputs=[],
        )
    progress.finish()

    frames = result.frames[0]
    save_video(frames, args.output)


if __name__ == "__main__":
    main()
