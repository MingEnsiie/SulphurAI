# SulphurAI

SulphurAI is a local inference wrapper for Sulphur 2 / LTX-2.3 text-to-video and image-to-video generation. It loads a local Sulphur checkpoint, builds the Diffusers LTX2 pipeline, and writes generated video frames to an MP4 file.

## Features

- Text-to-video generation from a prompt.
- Image-to-video generation from a prompt and input image.
- Local Sulphur 2 checkpoint loading from `sulphur_dev_bf16.safetensors`.
- Optional local Gemma 3 12B text encoder path.
- MP4 output via `imageio[ffmpeg]`.

## Requirements

- Linux with a CUDA-capable NVIDIA GPU.
- Python 3.12 recommended.
- CUDA-compatible PyTorch.
- Local Sulphur model assets.

By default, the loader looks for model assets next to this repository:

```text
../Assets/models/SulphurAI-Sulphur-2-base/sulphur_dev_bf16.safetensors
../Assets/models/gemma-3-12b-it-bnb-4bit
```

You can override the defaults with environment variables:

```bash
export SULPHUR_MODELS_DIR=/path/to/Assets/models
export SULPHUR_CHECKPOINT=/path/to/sulphur_dev_bf16.safetensors
export SULPHUR_TEXT_ENCODER=/path/to/gemma-3-12b-it-bnb-4bit
```

If the local Gemma text encoder is missing, the script falls back to `unsloth/gemma-3-12b-it-bnb-4bit`.

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .sulphur
source .sulphur/bin/activate
pip install -r requirements.txt
```

This project already uses `.sulphur` as the local virtual environment directory.

## Usage

Text-to-video:

```bash
.sulphur/bin/python generate.py \
  --prompt "a cinematic shot of a cat running on a beach" \
  --output outputs/cat_beach.mp4
```

During generation, the CLI prints a progress bar with elapsed time and estimated time remaining. It also shows progress while writing the MP4 file.

Image-to-video:

```bash
.sulphur/bin/python generate.py \
  --prompt "camera slowly zooms in, soft morning light" \
  --image input.jpg \
  --output outputs/zoom.mp4
```

Use a custom text encoder path:

```bash
.sulphur/bin/python generate.py \
  --prompt "a spaceship crossing a nebula" \
  --text-encoder /path/to/gemma-3-12b \
  --output outputs/spaceship.mp4
```

Show all options:

```bash
.sulphur/bin/python generate.py --help
```

## Main Options

| Option | Default | Description |
| --- | --- | --- |
| `--prompt` | required | Positive generation prompt. |
| `--negative-prompt` | `""` | Negative prompt. |
| `--image` | unset | Enables image-to-video mode with an input image. |
| `--width` | `768` | Output width. |
| `--height` | `512` | Output height. |
| `--frames` | `65` | Number of frames. Must be `8n + 1`. |
| `--steps` | `40` | Inference steps. |
| `--guidance-scale` | `3.5` | Classifier-free guidance scale. |
| `--seed` | `42` | Random seed. |
| `--output` | `outputs/output.mp4` | MP4 output path. |
| `--text-encoder` | auto | Local path or Hugging Face repo for Gemma 3 12B. |

## Project Structure

```text
generate.py              CLI entry point
src/pipeline.py          Sulphur 2 / LTX2 pipeline loader
scripts/watch_download.py Gemma model download progress helper
requirements.txt         Python dependencies
outputs/                 Generated videos
```

## Notes

- Generated videos are written under `outputs/` by default.
- The model files are large and are not stored in this repository.
- The pipeline currently targets CUDA by default and uses `torch.bfloat16`.
- For best results, run from the repository root so relative output paths resolve correctly.
