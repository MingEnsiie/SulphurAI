# SulphurAI Project

## Model Paths

All models are stored locally at:

```
/home/mingzhang/Downloads/code/Assets/models/
```

Key models:
- `SulphurAI-Sulphur-2-base/sulphur_dev_bf16.safetensors` — main 43GB checkpoint (BF16)
- `gemma-3-12b-it-bnb-4bit/` — text encoder (Gemma 3 12B, BNB 4-bit quantized)

## Architecture

Sulphur 2 is LTX-2.3 (audiovisual text-to-video). The checkpoint uses Lightricks-internal key
names; `src/pipeline.py` remaps them to diffusers format at load time.

Key configs differ from the public LTX-2 model:
- Encoder: max 1024 channels (LTX-2 public uses 2048)
- Decoder: 4 up-blocks with mixed upsample factors
- Text encoder: Gemma 3 12B (hidden_size=5376); connectors use `text_proj_in_factor=35`
- All gated attention enabled; per-modality projections enabled

## Running

```bash
cd /home/mingzhang/Downloads/code/SulphurAI
source .sulphur/bin/activate  # activate virtualenv

python generate.py \
  --prompt "a cat playing piano" \
  --width 512 --height 288 --frames 49 \
  --steps 30 --guidance-scale 5.0 \
  --output outputs/test.mp4
```

## Do NOT

- Download from HuggingFace — all models are already local
- Use `_LTX2_CACHE` or any HF hub path in code
- Change model config without re-verifying key counts (run the verify script in `src/pipeline.py`)
