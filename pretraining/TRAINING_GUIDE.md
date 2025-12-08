# CLIP 预训练/微调指南

## 📋 训练策略说明

本项目支持**分层训练**策略，可以只训练CLIP模型的特定部分，实现高效的预训练/微调。

### 🎯 核心概念

CLIP模型包含以下主要组件：
```
CLIP Model (151M 参数)
├── Vision Encoder (视觉编码器, ~85M 参数)
│   └── ViT-Base-Patch32
├── Text Encoder (文本编码器, ~63M 参数)
│   └── Transformer
├── Visual Projection (视觉投影层, ~0.6M 参数)
│   └── Linear: 768 → 512
└── Text Projection (文本投影层, ~0.4M 参数)
    └── Linear: 512 → 512
```

### 🔒 冻结策略 (--freeze_strategy)

| 策略名称 | 训练的部分 | 冻结的部分 | 参数量 | 适用场景 |
|---------|-----------|-----------|--------|---------|
| `none` | 全模型 | 无 | 151M (100%) | 完整预训练，数据量大 |
| `projection_only` | 投影层 | 编码器 | ~1M (~0.7%) | **推荐**：快速适配，数据量小 |
| `encoders` | 投影层 | 两个编码器 | ~1M (~0.7%) | 同 projection_only |
| `vision_encoder` | 文本编码器+投影层 | 视觉编码器 | ~64M (~42%) | 文本域适配 |
| `text_encoder` | 视觉编码器+投影层 | 文本编码器 | ~86M (~57%) | 视觉域适配 |

---

## 🚀 完整训练命令及参数说明

### 1. **推荐配置：只训练投影层（projection_only）**

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

**参数详解：**

- `--dataset masked`
  - **作用**：选择数据集类型
  - **可选值**：
    - `original`：原始图片（data/img/，约10000张）
    - `masked`：Mask图片（data/sample_data/top_800_mask/，800张）
    - `inpainted`：Inpaint图片（data/sample_data/top_800_inpaint/，800张）
  - **说明**：masked/inpainted 自动使用 `top_800_info.csv`（800张完整数据集）

- `--image_pair text`
  - **作用**：选择配对的文本类型
  - **可选值**：
    - `text`：使用原始文本（来自数据集）
    - `caption`：使用生成的caption（需要先运行caption生成）
  - **推荐**：`text`（直接可用）

- `--freeze_strategy projection_only`
  - **作用**：冻结策略，控制哪些层可训练
  - **可选值**：见上表
  - **推荐**：`projection_only`（只训练1M参数，速度快，过拟合风险低）

- `--gpus 0`
  - **作用**：指定使用的GPU
  - **格式**：单卡用 `0`，多卡用 `"0 1 2"`（空格分隔）
  - **当前**：RTX 4070 Laptop GPU

- `--batch_size 32`
  - **作用**：批次大小
  - **推荐值**：
    - `projection_only` 策略：32-64（显存占用少）
    - `none` 全模型训练：8-16（显存占用大）
  - **说明**：RTX 4070 支持较大batch size

- `--lr 5e-4`
  - **作用**：学习率
  - **推荐值**：
    - `projection_only`：5e-4 到 1e-3（投影层可以用较大学习率）
    - `none` 全模型：1e-4 到 5e-5（编码器需要小学习率）

- `--max_epochs 20`
  - **作用**：最大训练轮数
  - **推荐值**：
    - `projection_only`：10-30 epochs（收敛快）
    - `none` 全模型：5-10 epochs（收敛慢）

- `--log_every_n_steps 10`
  - **作用**：每N步打印一次日志
  - **说明**：用于监控训练进度

---

### 2. **快速测试配置（10%数据）**

```powershell
C:/Users/Aiur/miniconda3/envs/hateful-image-ofa/python.exe pretraining/main.py `
  --dataset masked `
  --image_pair text `
  --freeze_strategy projection_only `
  --gpus 0 `
  --batch_size 32 `
  --lr 5e-4 `
  --max_epochs 5 `
  --limit_train_batches 0.1 `
  --limit_val_batches 0.5 `
  --log_every_n_steps 3
```

**额外参数：**

- `--limit_train_batches 0.1`
  - **作用**：只使用10%的训练数据
  - **范围**：0.0 到 1.0
  - **用途**：快速验证代码、超参数搜索

- `--limit_val_batches 0.5`
  - **作用**：只使用50%的验证数据
  - **范围**：0.0 到 1.0
  - **注意**：验证集小时需要提高比例（如0.5）避免批次不足错误

---

### 3. **全模型训练配置（完整预训练）**

```powershell
C:/Users/Aiur/miniconda3/envs/hateful-image-ofa/python.exe pretraining/main.py `
  --dataset masked `
  --image_pair text `
  --freeze_strategy none `
  --gpus 0 `
  --batch_size 16 `
  --lr 1e-4 `
  --max_epochs 10 `
  --log_every_n_steps 20
```

---

## 📊 数据集信息

### Top 800 数据集 (masked/inpainted)
```
总计: 800 张图片
├── 训练集 (train): 659 张 (82.4%)
├── 验证集 (dev): 51 张 (6.4%)
└── 测试集 (test): 90 张 (11.2%)
```

### 训练统计（800张完整数据）
- Batch size = 32
  - 训练批次数：659 ÷ 32 ≈ 21 batches/epoch
  - 验证批次数：51 ÷ 32 ≈ 2 batches/epoch
  
- Batch size = 16
  - 训练批次数：659 ÷ 16 ≈ 42 batches/epoch
  - 验证批次数：51 ÷ 16 ≈ 4 batches/epoch

---

## 💾 模型保存

### Checkpoint 位置
```
pretraining/checkpoints/model-epoch=XX.ckpt
```

### 自动保存策略
- 监控指标：`val/loss`（验证集损失）
- 保存规则：只保存最佳模型（val_loss最小）
- 格式：PyTorch Lightning checkpoint（包含模型权重、优化器状态等）

### 加载预训练的投影层

训练完成后，可以在主训练脚本中加载投影层：

```python
import torch

# 加载 checkpoint
checkpoint = torch.load('pretraining/checkpoints/model-epoch=10.ckpt')

# 提取投影层权重
clip_state = checkpoint['state_dict']
projection_weights = {
    k.replace('clip.', ''): v 
    for k, v in clip_state.items() 
    if 'projection' in k
}

# 在主模型中加载
model.clip.load_state_dict(projection_weights, strict=False)
```

---

## 🎓 训练流程推荐

### 两阶段训练策略

**阶段1：投影层预训练（本脚本）**
```bash
# 在 Hateful Memes 数据上只训练投影层
python pretraining/main.py \
  --dataset masked \
  --freeze_strategy projection_only \
  --batch_size 32 \
  --lr 5e-4 \
  --max_epochs 20
```

**阶段2：全模型微调（主训练脚本）**
```bash
# 加载预训练的投影层，继续在 Hateful Memes 上微调全模型
python main_training.py \
  --load_projection pretraining/checkpoints/model-epoch=19.ckpt \
  --freeze_strategy none \
  --batch_size 16 \
  --lr 1e-4 \
  --max_epochs 10
```

**优势：**
1. ✅ 投影层先适配 Hateful Memes 任务
2. ✅ 参数高效（阶段1只训练1M参数）
3. ✅ 减少过拟合风险
4. ✅ 加速收敛

---

## 📈 预期训练效果

### Projection Only 策略
- **参数量**：~1M / 151M (~0.7%)
- **训练速度**：~5-6 it/s (RTX 4070)
- **显存占用**：~4-6 GB (batch_size=32)
- **收敛速度**：5-10 epochs
- **Val Loss**：预期从 ~3.0 降至 ~2.5

### Full Model 策略
- **参数量**：151M (100%)
- **训练速度**：~4-5 it/s (RTX 4070)
- **显存占用**：~8-10 GB (batch_size=16)
- **收敛速度**：10-20 epochs
- **Val Loss**：预期从 ~3.0 降至 ~2.2

---

## ⚠️ 常见问题

### Q1: MisconfigurationException: limit_val_batches * num_batches < 1
**原因**：验证批次太少（如 51÷32=2 batches，0.1*2=0.2<1）
**解决**：提高 `--limit_val_batches` 到 0.5 或更高

### Q2: GPU 显存不足
**解决**：
1. 减小 `--batch_size`（如从32改为16）
2. 使用 `projection_only` 策略（显存占用更少）

### Q3: 训练太慢
**解决**：
1. 使用 `projection_only` 策略（速度提升20-30%）
2. 增大 `--batch_size`（提高GPU利用率）
3. 减小数据集（使用 `--limit_train_batches 0.5`）

### Q4: 如何查看可训练参数？
**方法**：训练开始时会自动打印
```
🔒 冻结策略: 只训练投影层 (visual_projection + text_projection)
📊 可训练参数: 1,049,088 / 151,277,313 (0.69%)
```

---

## 📝 其他参数

```bash
--image_size 224           # 图片大小（CLIP固定224）
--weight_decay 1e-4        # 权重衰减（正则化）
--gradient_clip_val 0.1    # 梯度裁剪（防止梯度爆炸）
--max_steps -1             # 最大训练步数（-1=使用epochs）
--val_check_interval 1.0   # 验证频率（1.0=每epoch验证一次）
--strategy None            # 分布式策略（单卡不需要）
--clip_pretrained_model openai/clip-vit-base-patch32  # CLIP预训练模型
```

---

## 🔍 监控训练

### 查看日志
```bash
# 训练过程会实时打印
Epoch 0: 100%|██| 21/21 [00:04<00:00, 4.5it/s, loss=2.89]
Epoch 0, global step 21: 'val/loss' reached 2.87 (best 2.87)
```

### WandB 集成（可选）
如果配置了 WandB，会自动上传训练指标到云端：
- Loss 曲线
- 学习率变化
- 参数更新统计

---

## 🎯 总结

**最推荐的工作流：**

1. **快速验证**（5分钟）
   ```bash
   --freeze_strategy projection_only --max_epochs 3 --limit_train_batches 0.1
   ```

2. **投影层预训练**（30分钟）
   ```bash
   --freeze_strategy projection_only --max_epochs 20 --batch_size 32
   ```

3. **保存最佳模型**
   - 自动保存到 `checkpoints/model-epoch=XX.ckpt`

4. **加载到主训练脚本**
   - 用预训练的投影层初始化主模型
   - 继续全模型微调

**优势**：参数高效、训练快速、效果更好！🚀
