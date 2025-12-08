# 完整训练命令说明

## ✅ 你的命令会使用GPU！

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

## 📊 训练配置详解

### 数据集配置
- **`--dataset masked`**
  - 使用 `data/sample_data/top_800_mask/` 文件夹
  - 共 800 张 masked 图片
  - 自动读取 `top_800_info.csv` 元数据
  - 数据划分：
    - 训练集: 659 张 (82.4%)
    - 验证集: 51 张 (6.4%)
    - 测试集: 90 张 (11.2%)

- **`--image_pair text`**
  - 使用数据集原始文本（来自 CSV 的 text 列）
  - 不使用生成的 caption

### 模型配置
- **`--freeze_strategy projection_only`** ⭐ 核心设置
  - **冻结**: Vision Encoder (85M) + Text Encoder (63M)
  - **训练**: Visual Projection (0.6M) + Text Projection (0.4M)
  - **总可训练参数**: ~1,049,088 / 151,277,313 ≈ **0.7%**
  - **优势**:
    - ✅ 训练速度快（只更新1M参数）
    - ✅ 显存占用低（可以用更大batch size）
    - ✅ 过拟合风险小（参数少）
    - ✅ 适合小数据集（800张）

### GPU 配置
- **`--gpus 0`** ✅ 使用GPU
  - 设备: NVIDIA GeForce RTX 4070 Laptop GPU
  - CUDA 版本: 11.8
  - PyTorch: 2.0.1+cu118
  - 代码会自动检测并使用GPU
  - 如果CUDA不可用，会自动降级到CPU

### 训练超参数
- **`--batch_size 32`**
  - 每批处理 32 张图片
  - projection_only 模式显存占用低，可以用大batch size
  - 每 epoch 训练批次: 659 ÷ 32 ≈ **21 batches**
  - 每 epoch 验证批次: 51 ÷ 32 ≈ **2 batches**

- **`--lr 5e-4`** (0.0005)
  - 学习率
  - projection_only 可以用较大学习率（投影层从头训练）
  - 全模型训练通常用 1e-4 (0.0001)

- **`--max_epochs 20`**
  - 最多训练 20 轮
  - 每轮约 21 batches × 2 steps = 42 steps
  - 总训练步数: 20 × 21 ≈ **420 steps**
  - 预计耗时: ~30-40 分钟（GPU模式）

- **`--log_every_n_steps 10`**
  - 每 10 步打印一次日志
  - 用于监控训练进度

---

## 🎯 预期训练效果

### 训练过程
```
🔒 冻结策略: 只训练投影层 (visual_projection + text_projection)
📊 可训练参数: 1,049,088 / 151,277,313 (0.69%)

Found 659 images with matching CSV entries for split=train
Found 51 images with matching CSV entries for split=dev

✓ Using GPU: [0]
NVIDIA GeForce RTX 4070 Laptop GPU

Epoch 0: 100%|████| 21/21 [00:04<00:00, 5.2it/s, loss=2.85]
Epoch 0, global step 21: 'val/loss' reached 2.91 (best 2.91)

Epoch 10: 100%|████| 21/21 [00:03<00:00, 5.8it/s, loss=2.45]
Epoch 10, global step 210: 'val/loss' reached 2.52 (best 2.50)

Epoch 19: 100%|████| 21/21 [00:03<00:00, 5.9it/s, loss=2.30]
Epoch 19, global step 399: 'val/loss' reached 2.48 (best 2.48)
```

### 性能指标
- **训练速度**: ~5-6 it/s（每秒处理5-6个batch）
- **显存占用**: ~4-6 GB
- **每 epoch 耗时**: ~4-5 秒
- **总耗时**: 20 epochs × 4秒 ≈ **1.5-2 分钟**（纯训练时间）
  - 加上数据加载、验证等，实际约 **5-8 分钟**
- **预期 val_loss**: 
  - 初始: ~2.9-3.0
  - 最终: ~2.4-2.5

### 保存的模型
- **位置**: `pretraining/checkpoints/model-epoch=XX.ckpt`
- **保存策略**: 只保存 val_loss 最低的模型
- **文件大小**: ~600 MB（包含完整CLIP模型权重）

---

## 🚀 开始训练

### 方式1: 直接运行命令
```powershell
C:/Users/Aiur/miniconda3/envs/hateful-image-ofa/python.exe pretraining/main.py --dataset masked --image_pair text --freeze_strategy projection_only --gpus 0 --batch_size 32 --lr 5e-4 --max_epochs 20 --log_every_n_steps 10
```

### 方式2: 使用脚本
```powershell
# 完整训练（20 epochs）
powershell -ExecutionPolicy Bypass -File pretraining/run_projection_pretrain.ps1

# 快速测试（3 epochs，10%数据）
powershell -ExecutionPolicy Bypass -File pretraining/run_quick_test.ps1

# 检查GPU状态
powershell -ExecutionPolicy Bypass -File pretraining/check_gpu.ps1
```

---

## 📌 重要提示

### GPU vs CPU 对比
| 配置 | 命令 | 速度 | 耗时 (20 epochs) |
|------|------|------|-----------------|
| **GPU** (推荐) | `--gpus 0` | 5-6 it/s | ~5-8 分钟 |
| CPU | `--gpus ''` | 1-2 it/s | ~20-30 分钟 |

### 其他数据集
```bash
# 使用 inpainted 图片（同样800张）
--dataset inpainted

# 使用原始图片（约10000张，需要更多时间）
--dataset original
```

### 调整batch size
- RTX 4070 推荐: 32-64 (projection_only)
- 显存不足时: 降到 16 或 8
- 全模型训练: 8-16

---

## ✅ 确认信息

**你的命令配置：**
- ✅ 使用 GPU (RTX 4070)
- ✅ 800张完整数据集 (masked)
- ✅ 只训练投影层 (1M参数)
- ✅ 参数设置合理
- ✅ 预计 5-8 分钟完成

**开始训练吧！** 🚀
