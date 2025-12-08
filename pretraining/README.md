# Pretraining 使用说明

## 📚 完整文档

详细的训练策略和参数说明请查看：**[TRAINING_GUIDE.md](./TRAINING_GUIDE.md)**

---

## 🚀 快速开始

### 推荐：投影层预训练（只训练1M参数）

**为什么选择投影层预训练？**
- ✅ 参数少（1M vs 151M），训练快
- ✅ 显存占用低，支持大batch size
- ✅ 过拟合风险小
- ✅ 可以作为主训练的初始化

**一键启动：**
```powershell
# 完整训练（800张图片，20 epochs，约30分钟）
powershell -ExecutionPolicy Bypass -File pretraining/run_projection_pretrain.ps1

# 快速测试（10%数据，3 epochs，约2分钟）
powershell -ExecutionPolicy Bypass -File pretraining/run_quick_test.ps1
```

**手动命令：**
```powershell
C:/Users/Aiur/miniconda3/envs/hateful-image-ofa/python.exe pretraining/main.py `
  --dataset masked `
  --image_pair text `
  --freeze_strategy projection_only `
  --gpus 0 `
  --batch_size 32 `
  --lr 5e-4 `
  --max_epochs 20 `
  --log_every_n_steps 10
```

---

## 🎯 训练策略对比

| 策略 | 可训练参数 | 训练速度 | 显存占用 | 推荐场景 |
|------|-----------|---------|---------|---------|
| `projection_only` | 1M (0.7%) | 快 (~5-6 it/s) | 低 (~4-6GB) | ✅ **推荐** 预训练/快速适配 |
| `none` (全模型) | 151M (100%) | 慢 (~4-5 it/s) | 高 (~8-10GB) | 完整预训练，数据量大 |
| `vision_encoder` | 64M (42%) | 中等 | 中等 | 文本域适配 |
| `text_encoder` | 86M (57%) | 中等 | 中等 | 视觉域适配 |

---

## 环境要求
- Python 3.9
- PyTorch 2.0.1+cu118 (CUDA 11.8) ✓ 已安装
- GPU: NVIDIA GeForce RTX 4070 Laptop GPU ✓ 可用
- 已安装：transformers, pytorch-lightning<2, pillow, pandas, wandb

## 快速开始

### GPU 模式运行（推荐，已启用）

**方式1：使用脚本（推荐）**
```powershell
# 投影层预训练（800张完整数据）
powershell -ExecutionPolicy Bypass -File pretraining/run_projection_pretrain.ps1

# 快速测试（10%数据验证）
powershell -ExecutionPolicy Bypass -File pretraining/run_quick_test.ps1
```

**方式2：手动命令**
```powershell
# 投影层预训练（只训练1M参数）
C:/Users/Aiur/miniconda3/envs/hateful-image-ofa/python.exe pretraining/main.py `
  --dataset masked `
  --image_pair text `
  --freeze_strategy projection_only `
  --gpus 0 `
  --batch_size 32 `
  --lr 5e-4 `
  --max_epochs 20 `
  --log_every_n_steps 10

# 全模型训练（训练151M参数）
C:/Users/Aiur/miniconda3/envs/hateful-image-ofa/python.exe pretraining/main.py `
  --dataset masked `
  --image_pair text `
  --freeze_strategy none `
  --gpus 0 `
  --batch_size 16 `
  --lr 1e-4 `
  --max_epochs 10 `
  --log_every_n_steps 10
```

**训练效果（投影层预训练）：**
- ✓ 数据集: 800张 masked 图片（训练659，验证51，测试90）
- ✓ GPU: NVIDIA GeForce RTX 4070 Laptop GPU
- ✓ 可训练参数: ~1M / 151M (0.7%)
- ✓ 训练速度：~5-6 it/s
- ✓ 显存占用: ~4-6 GB
- ✓ 预期 val_loss: 3.0 → 2.5
- ✓ 自动保存最佳 checkpoint 到 `pretraining/checkpoints/`

---

### CPU 模式（备用）
如需在CPU上运行（不推荐，速度慢）：
```powershell
C:/Users/Aiur/miniconda3/envs/hateful-image-ofa/python.exe pretraining/main.py `
  --dataset masked `
  --image_pair text `
  --gpus 0 `
  --batch_size 8 `
  --lr 1e-4 `
  --max_epochs 2 `
  --limit_train_batches 0.02 `
  --limit_val_batches 0.15 `
  --log_every_n_steps 2
```

### CPU 模式（备用）
如需在CPU上运行（不推荐，速度慢）：
```powershell
C:/Users/Aiur/miniconda3/envs/hateful-image-ofa/python.exe pretraining/main.py `
  --dataset masked `
  --image_pair text `
  --gpus 0 `
  --batch_size 8 `
  --lr 1e-4 `
  --max_epochs 2 `
  --limit_train_batches 0.02 `
  --limit_val_batches 0.15 `
  --log_every_n_steps 2
```

---

## 参数说明

### 核心参数

**--freeze_strategy** （冻结策略）⭐ 新增
- `projection_only`: 只训练投影层（推荐，1M参数）
- `none`: 全模型训练（151M参数）
- `encoders`: 同projection_only
- `vision_encoder`: 冻结视觉编码器
- `text_encoder`: 冻结文本编码器

**--dataset** （数据集选择）
- `masked`: 使用 masked 图片（800张，自动用top_800_info.csv）
- `inpainted`: 使用 inpainted 图片（800张）
- `original`: 使用原始图片（约10000张）

**--image_pair** （文本配对）
- `text`: 使用原始文本（推荐）
- `caption`: 使用生成的caption

### 数据集选项
- `--dataset`: 选择数据集
  - `original`: 原始图片（`data/img/`）
  - `masked`: Mask 图片（`data/sample_data/top_800_mask/`）- 当前 659 张
  - `inpainted`: Inpaint 图片（`data/sample_data/top_800_inpaint/`）

- `--image_pair`: 配对文本类型
  - `text`: 使用原始文本
  - `caption`: 使用生成的 caption（需先运行 caption 生成）

### 训练参数
- `--gpus`: GPU 编号，多卡用空格分隔（如 `"0 1"`）
  - 当前系统：单卡 RTX 4070，使用 `--gpus 0`
- `--batch_size`: 批次大小
  - GPU (RTX 4070): 推荐 16-32
  - CPU: 推荐 4-8
- `--lr`: 学习率（默认 1e-4）
- `--max_epochs`: 最大训练轮数
- `--limit_train_batches`: 训练数据比例（0.1 = 10%）
- `--limit_val_batches`: 验证数据比例
- `--log_every_n_steps`: 每 N 步打印日志

---

## 输出说明

### Checkpoint 位置
训练过程自动保存最佳模型到：
```
pretraining/checkpoints/model-epoch=XX.ckpt
```

### 训练日志示例
```
✓ Using GPU: [0]
NVIDIA GeForce RTX 4070 Laptop GPU
Epoch 0: 100%|████| 6/6 [00:01<00:00, 4.11it/s, loss=2.89]
Epoch 0, global step 4: 'val/loss' reached 2.96836 (best 2.96836)
Epoch 4: 100%|████| 6/6 [00:01<00:00, 4.54it/s, loss=2.85]
Epoch 4, global step 20: 'val/loss' reached 2.77100 (best 2.77100)
```

---

## 常见问题

### Q: 如何查看GPU状态？
A: 运行以下命令：
```powershell
C:/Users/Aiur/miniconda3/envs/hateful-image-ofa/python.exe -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

### Q: 如何切换到完整数据集？
A: 去掉 `limit_train_batches` 和 `limit_val_batches` 参数即可

### Q: 如何使用原始图片而非 masked？
A: 将 `--dataset masked` 改为 `--dataset original`

### Q: GPU内存不足？
A: 减小 `batch_size`（如从 16 改为 8 或 4）
