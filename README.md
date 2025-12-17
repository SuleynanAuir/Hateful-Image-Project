## Hateful-Image-Project

  Dear Professor Raymond Lee:
  It is our project——a PyTorch Lightning pipeline for hateful memes detection on the Hateful Memes dataset and its inpainted/masked variants, using CLIP-based multimodal encoders.

---

## 1. Installation

### 1.1. Create environment (recommended: Conda)

```bash
conda create -n hateful-image python=3.9 -y
conda activate hateful-image
```

### 1.2. Install Python dependencies

From the project root (`Hateful-Image-Project/`):

```bash
pip install -r requirements.txt
```

This installs:
- **PyTorch / PyTorch Lightning** (training loop and GPU support)
- **Transformers / CLIP-related packages**
- **HuggingFace Datasets / data utilities**
- **wandb** (for experiment logging; can be disabled)

If you need the OFA-based preprocessing environment, see `README_OFA_ENV.md` (optional; only for caption generation/inpainting helpers in `data_preprocess/` and `OFA/`).

---

## 2. Environment & Data Preparation

### 2.1. GPU and CUDA

- The default training script assumes **GPU** is available.
- Make sure your installed PyTorch build matches your CUDA version (see the official PyTorch “Get Started” page).

### 2.2. Dataset layout

By default, the code expects data under `data/`:
- `data/all_data/` – full Hateful Memes JSON/CSV files.
- `data/sample_data/` – smaller sample subsets (e.g. top_800, top_10000).
- `data/img/` – original meme images.

The new training pipeline uses the dataset name to select which view of the data to load:
- `original` – original memes.
- `masked` – text-masked images (see `data/sample_data/top_800_mask/` etc.).
- `inpainted` – inpainted images (see `data/sample_data/top_800_inpaint/` etc.).

Make sure the paths and CSV/JSON files match what `datasets.py` / `datasets_new.py` expect.

---

## 3. Main Training Script (`new_main_enhanced.py`)

The recommended entry point for the new pipeline is:

```bash
python new_main_enhanced.py [ARGS...]
```

### 3.1. Key arguments

Below are the most important CLI arguments (see `new_main_enhanced.py` for the full list and defaults):

- `--dataset`  
  Choices: `original`, `masked`, `inpainted`  
  Controls which image variant to load.

- `--labels`  
  Default: `original`  
  Type of labels to use (e.g., binary original labels).

- `--image_size`  
  Default: `224`  
  Input resolution for the CLIP image encoder.

- `--clip_pretrained_model`  
  Default: `openai/clip-vit-base-patch32`  
  HuggingFace model name for the CLIP backbone.

- `--caption_mode`  
  Choices: `none`, `replace_text`, `concat_with_text`  
  How to use captions with the meme text, if available.

- `--multilingual_tokenizer_path`  
  Path to a multilingual tokenizer (or `none` to disable).

- `--freeze_image_encoder` / `--freeze_text_encoder`  
  `True`/`False` flags. When `True`, CLIP encoders are frozen and only projection + classifier layers are trained.

- `--batch_size`  
  Training batch size. Default: `32`.

- `--lr` / `--weight_decay`  
  Optimizer hyperparameters.

- `--max_epochs`  
  Number of training epochs.

- `--gpus`  
  GPU index (e.g. `0`) or string understood by PyTorch Lightning’s `Trainer`.

- `--limit_train_batches`, `--limit_val_batches`  
  Fractions in \([0,1]\) to speed up experiments by subsampling batches.

- `--use_pretrained_map` / `--local_pretrained_weights`  
  Whether to load a **projection head** from a pretraining checkpoint (e.g. from `pretraining/`).

- `--wandb_mode`  
  Choices: `offline`, `online`, `disabled`  
  - `online`: log to your WandB project (requires `wandb login`).  
  - `offline`: log locally to `wandb/` for later sync.  
  - `disabled`: do not use WandB.

- `--wandb_timeout`, `--wandb_base_url`  
  Advanced options for WandB initialization.

---

## 4. Basic Usage (We provide a Command-Line Examples)

### 4.1. Example: Inpainted dataset, binary labels (recommended starting point)

From the project root:

```bash
python new_main_enhanced.py \
  --dataset inpainted \
  --labels original \
  --freeze_image_encoder True \
  --freeze_text_encoder True \
  --batch_size 16 \
  --lr 1e-5 \
  --weight_decay 1e-4 \
  --max_epochs 1 \
  --log_every_n_steps 10 \
  --gpus 0 \
  --wandb_mode offline
```

This will:
- Load train/validation/test splits via `load_dataset(...)`.
- Use frozen CLIP encoders and train only the projection + classifier.
- Log metrics to local WandB files (in `wandb/`).
- Save checkpoints to `checkpoints_new/`.

### 4.2. Example: Original images, offline WandB logging

```bash

python new_main_enhanced.py \
  --dataset original \
  --labels original \
  --freeze_image_encoder True \
  --freeze_text_encoder True \
  --batch_size 16 \
  --lr 1e-4 \
  --max_epochs 20 \
  --gpus 0 \
  --wandb_mode offline
```

### 4.3. Example: Using a pretrained projection head

Assuming you have run pretraining in `pretraining/` and produced a checkpoint:

```bash
python new_main_enhanced.py \
  --dataset inpainted \
  --labels original \
  --freeze_image_encoder True \
  --freeze_text_encoder True \
  --use_pretrained_map True \
  --local_pretrained_weights pretraining/checkpoints/model-epoch=19.ckpt \
  --batch_size 16 \
  --lr 1e-5 \
  --max_epochs 5 \
  --gpus 0 \
  --wandb_mode offline
```


---

## 5. Other Entry Points and Guides

- `main.py`  
  It is our elder training pipeline (used by `SAMPLE800_TRAINING*.md` and `run_sample800_*.ps1`).  
  For new experiments, **prefer** `new_main_enhanced.py` unless you need strict backward compatibility. And it is our mainly use.

- `pretraining/`  
  Utilities for representation / projection pretraining. See `pretraining/README.md` and `pretraining/TRAINING_GUIDE.md`.

- `OFA/` and `data_preprocess/`  
  OFA-based captioning and inpainting helpers. See `README_OFA_ENV.md` for environment setup if you plan to regenerate captions or masks.



