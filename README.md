# Reason Through the Latent!

Official implementation of **Causal Visual Recurrent Reasoning (CVRR)**.

## Overview

Latent visual states can retain task-relevant information without being used by
the answer decoder. CVRR is designed to make that visual information causally
necessary while preserving the visual competence of a pretrained VLM.

CVRR uses three components:

1. **Causal Visual-Read Boundary:** layer-wise activation patching identifies
   where visual information has been incorporated into task-relevant question
   representations.
2. **Persistent Visual Recurrence:** the decoder layer immediately after the
   boundary is reused as a shared recurrent transition. The question state is
   updated repeatedly while the native visual rows remain available inside the
   recurrent computation.
3. **Strict Causal Decoder Interface:** only the final recurrent question state
   reaches the upper answer decoder. Original multimodal hidden states, visual
   rows, and multimodal prefix caches are excluded from answer decoding.

For Qwen2.5-VL-7B, the default configuration uses boundary layer 20, decoder
layer 21 as the shared transition, four recurrent states, and rank-32 LoRA on
the transition layer's attention and MLP projections. The dense backbone stays
frozen, and training uses answer-token cross entropy only.

```text
Image + Question
      |
Frozen layers 0--20
      |
Native multimodal initialization
      |
Shared layer-21 recurrent transition x T
      |
Final recurrent question state
      |
Frozen layers 22--27
      |
Answer
```

## Installation

We recommend Python 3.10 or newer with a CUDA-enabled PyTorch installation.

```bash
cd CVRR
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The reference environment uses Python 3.11, PyTorch 2.9.1, Transformers
4.57.6, and PEFT 0.17.1. Base weights and processors are loaded from the local
Hugging Face cache or downloaded when network access is available.

## Implementation

### Step 1: Prepare Visual-CoT Data

Materialize the training data as either a Hugging Face
`Dataset.save_to_disk` directory or the supported sharded manifest layout.
Each record must contain:

- `image_bytes` or `image`
- `fixed_question`
- `fixed_answer`
- `fixed_hint` (optional)

Multiple-choice formatting should be fixed before the train/validation split
and stored in these fields. Set the resulting path in
[`configs/train_cvrr_qwen25_7b.yaml`](configs/train_cvrr_qwen25_7b.yaml).

### Step 2: Train CVRR

Single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash scripts/train.sh configs/train_cvrr_qwen25_7b.yaml
```

Multi-GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
bash scripts/train.sh configs/train_cvrr_qwen25_7b.yaml
```

The launcher infers one process per visible device when `NPROC_PER_NODE` is
not specified. The reference recipe uses per-device batch size 16 and gradient
accumulation 2 on four GPUs, giving an effective batch size of 128.

To select a different output directory or resume training:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/train.sh configs/train_cvrr_qwen25_7b.yaml \
  --output-dir outputs/cvrr-qwen2.5-vl-7b \
  --resume-from-checkpoint /path/to/checkpoint
```

### Step 3: Evaluate CVRR

Set the checkpoint and output paths in
[`configs/eval_cvrr.yaml`](configs/eval_cvrr.yaml), then run:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash scripts/eval.sh configs/eval_cvrr.yaml
```

Evaluation can be divided into deterministic shards:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash scripts/eval.sh configs/eval_cvrr.yaml \
  --shard 0 \
  --num-shards 4 \
  --output-dir predictions/shard-0
```

The evaluator saves raw JSONL predictions and aggregate JSON summaries.

### Step 4: Run Causal Analyses

All released analyses use one command dispatcher:

```bash
bash scripts/analyze.sh --help
```

The available commands cover boundary localization, causal intervention,
recurrence controls, learned-transition diagnostics, and efficiency. See
[`scripts/analysis_cvrr/README.md`](scripts/analysis_cvrr/README.md) for exact
inputs and commands.

### Step 5: Export a Hugging Face Checkpoint

Convert a full training checkpoint into a clean Hugging Face directory:

```bash
python convert_checkpoint.py \
  --checkpoint /path/to/training-checkpoint \
  --output /path/to/cvrr-hf
```

The converter validates the expected trainable tensors, preserves model
weights exactly, copies the processor, and writes the custom configuration and
modeling files required by `trust_remote_code=True`.

## Hugging Face Inference

An exported checkpoint supports `save_pretrained` and `from_pretrained`:

```python
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

checkpoint = "organization/cvrr-qwen2.5-vl-7b"
processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    checkpoint,
    trust_remote_code=True,
    dtype=torch.bfloat16,
).to("cuda").eval()

image = Image.open("example.jpg").convert("RGB")
question = "Which option is correct?\n(A) ...\n(B) ..."

multimodal_prompt = processor.apply_chat_template(
    [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": question},
        ],
    }],
    tokenize=False,
    add_generation_prompt=True,
)
text_prompt = processor.apply_chat_template(
    [{
        "role": "user",
        "content": [{"type": "text", "text": question}],
    }],
    tokenize=False,
    add_generation_prompt=True,
)

multimodal = processor(
    text=[multimodal_prompt],
    images=[image],
    return_tensors="pt",
)
text_only = processor.tokenizer(
    text_prompt,
    return_tensors="pt",
    add_special_tokens=False,
)
multimodal = {key: value.to(model.device) for key, value in multimodal.items()}
text_only = {key: value.to(model.device) for key, value in text_only.items()}
multimodal["pixel_values"] = multimodal["pixel_values"].to(model.dtype)

tokens = model.generate(
    **multimodal,
    question_ids=text_only["input_ids"],
    question_attention_mask=text_only["attention_mask"],
    do_sample=False,
    max_new_tokens=32,
)
answer = processor.tokenizer.decode(tokens[0], skip_special_tokens=True)
print(answer)
```

CVRR takes aligned multimodal and text-only encodings of the same prompt. The
text-only branch supplies the lower-layer answer-prefix context, while the
multimodal branch supplies the recurrent visual state.

## Evaluation

The common evaluator supports:

- V*
- MMVP
- BLINK
- MME-RealWorld-Lite

Two prediction protocols are implemented:

- `greedy`: deterministic answer generation with a shared token budget.
- `choice_logits`: restricted first-option-token scoring for causal analyses.

These protocols measure different quantities and should not be mixed in one
comparison. The evaluator applies a common visual-token ceiling of 8,192 and a
shared answer parser across the supported benchmarks.

## Repository Structure

```text
CVRR/
├── configs/
│   ├── train_cvrr_qwen25_7b.yaml   # Default training recipe
│   └── eval_cvrr.yaml              # Common benchmark evaluation recipe
├── cvrr/
│   ├── benchmarks.py               # Benchmark loading, prompts, and scoring
│   ├── configuration_cvrr.py       # Hugging Face CVRR configuration
│   ├── data.py                     # Visual-CoT loader and collator
│   └── modeling_cvrr.py            # Strict recurrent model implementation
├── scripts/
│   ├── analysis_cvrr/              # Causal and mechanistic analyses
│   │   ├── causal/                 # State interventions
│   │   ├── core/                   # Shared runtime, data, and statistics
│   │   ├── diagnostics/            # Transition diagnostics
│   │   └── recurrence/             # Recurrent controls and ablations
│   ├── analyze.sh                  # Analysis dispatcher
│   ├── eval.sh                     # Evaluation launcher
│   └── train.sh                    # Single- or multi-GPU training launcher
├── tests/                          # CPU structural and forward tests
├── convert_checkpoint.py           # Hugging Face checkpoint exporter
├── evaluate.py                     # Common benchmark evaluator
├── train.py                        # Distributed SFT entry point
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Citation
```bibtex
@misc{park2026reasonlatentmakinglatent,
      title={Reason Through the Latent! Making Latent Visual Reasoning Necessary}, 
      author={Suhyeong Park and Junha Jung and Jaewoo Kang},
      year={2026},
      eprint={2609.06746},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2609.06746}, 
}
```
