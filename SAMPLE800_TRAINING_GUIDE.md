# Sample 800 训练指南

## 📊 数据集信息

使用 `data/sample_data/top_800_mask/` 中的 800 张 masked 图片进行训练：

```
数据来源: top_800_info.csv
总计: 800 张图片
├── 训练集 (train): 659 张 (82.4%)
├── 验证集 (dev): 51 张 (6.4%)  
└── 测试集 (test): 90 张 (11.2%)

图片路径: data/sample_data/top_800_mask/*_masked.png
CSV文件: data/sample_data/top_800_info.csv
```

---

## 🚀 快速开始

### 方式1: 使用脚本（推荐）

**快速测试（2 epochs，10%数据）**
```powershell
powershell -ExecutionPolicy Bypass -File run_sample800_test.ps1
```

**完整训练（10 epochs，全部数据）**
```powershell
powershell -ExecutionPolicy Bypass -File run_sample800_train.ps1
```

### 方式2: 手动命令

```powershell
C:/Users/Aiur/miniconda3/envs/hateful-image-ofa/python.exe main.py --dataset masked --labels original --freeze_image_encoder True --freeze_text_encoder True --fusion align --gpus 0 --batch_size 16 --lr 1e-4 --max_epochs 10 --log_every_n_steps 20
```

---

## 🎯 模型架构

```
输入 Meme (图片 + 文本)
         │
    ┌────┴────┐
    │         │
    ▼         ▼
Vision     Text
Encoder    Encoder
(冻结❄️)   (冻结❄️)
    │         │
    ▼         ▼
 Image      Text
  Map       Map
(训练🔥)   (训练🔥)
    │         │
    └────┬────┘
         │
         ▼
     Fusion
   (align 策略)
         │
         ▼
   Pre-Output
    (MLP)
         │
         ▼
Classification Head
    (Binary)
         │
         ▼
   Hateful/Non-hateful
```

---

## 📝 参数说明

### 核心参数

**--dataset masked**
- 使用 masked 图片（top_800_mask 文件夹）
- 自动加载 `top_800_info.csv`
- 可选: `inpainted` (使用 top_800_inpaint)

**--labels original**
- 二分类任务（hateful/non-hateful）
- 可选: `fine_grained` (12个细粒度类别)

**--freeze_image_encoder True**
- 冻结CLIP Vision Encoder
- 只训练映射层和分类头

**--freeze_text_encoder True**
- 冻结CLIP Text Encoder
- 只训练映射层和分类头

**--fusion align**
- 图文融合策略
- 可选值:
  - `align`: 元素级乘法（CLIP默认）
  - `concat`: 拼接
  - `cross`: 交叉注意力
  - `attention_m`: 注意力机制

### 训练参数

**--batch_size 16**
- 推荐: GPU 16-32, CPU 4-8

**--lr 1e-4**
- 学习率
- 冻结encoder时推荐 1e-4

**--max_epochs 10**
- 最大训练轮数
- 800张数据推荐 10-20 epochs

**--num_mapping_layers 1**
- 映射层数量
- 默认1层，可增加到2-3层

**--map_dim 768**
- 映射维度
- 与CLIP特征维度对齐

### 监控参数

**--log_every_n_steps 20**
- 每20步打印日志

监控指标（自动保存最佳）:
- `val/auroc`: 验证集AUROC（主要指标）
- `val/acc`: 验证集准确率
- `val/f1`: 验证集F1分数

---

## 📈 预期效果

### 训练统计
```
Batch size = 16:
- 训练批次: 659 ÷ 16 ≈ 42 batches/epoch
- 验证批次: 51 ÷ 16 ≈ 4 batches/epoch
- 每epoch耗时: ~30-40秒（GPU）
- 10 epochs 总耗时: ~5-7分钟
```

### 性能指标
```
随机基线 (Random):
- Accuracy: ~50%
- AUROC: ~0.50

预期效果 (CLIP + Frozen Encoder):
- Accuracy: 60-70%
- AUROC: 0.65-0.75
- F1: 0.60-0.70

SOTA (全模型微调):
- Accuracy: 70-80%
- AUROC: 0.75-0.85
```

---

## 💾 输出文件

### Checkpoints
```
位置: checkpoints/wandb-run-name-epoch=XX.ckpt
保存策略: 只保存 val/auroc 最高的模型
```

### WandB 日志
```
项目: meme-v2
监控指标:
- train/loss
- val/loss
- val/auroc (主要)
- val/acc
- val/f1
- val/precision
- val/recall
```

---

## 🔧 高级配置

### 1. 使用预训练的投影层

如果你已经运行了 `pretraining/` 的投影层预训练：

```powershell
C:/Users/Aiur/miniconda3/envs/hateful-image-ofa/python.exe main.py `
  --dataset masked `
  --local_pretrained_weights pretraining/checkpoints/model-epoch=19.ckpt `
  --use_pretrained_map True `
  --freeze_image_encoder True `
  --freeze_text_encoder True `
  --gpus 0 `
  --batch_size 16 `
  --lr 1e-4 `
  --max_epochs 10
```

### 2. 细粒度分类

训练12个细粒度类别（种族、性别、宗教等）：

```powershell
C:/Users/Aiur/miniconda3/envs/hateful-image-ofa/python.exe main.py `
  --dataset masked `
  --labels fine_grained `
  --weight_fine_grained_loss 1.0 `
  --freeze_image_encoder True `
  --freeze_text_encoder True `
  --gpus 0 `
  --batch_size 16 `
  --lr 1e-4 `
  --max_epochs 15
```

### 3. 多任务学习

同时训练图像分类头和文本分类头：

```powershell
C:/Users/Aiur/miniconda3/envs/hateful-image-ofa/python.exe main.py `
  --dataset masked `
  --weight_image_loss 0.5 `
  --weight_text_loss 0.5 `
  --freeze_image_encoder True `
  --freeze_text_encoder True `
  --gpus 0 `
  --batch_size 16 `
  --lr 1e-4 `
  --max_epochs 10
```

---

## 🐛 常见问题

### Q1: FileNotFoundError: No such file or directory
**原因**: 图片文件路径不对
**检查**: 确认 `data/sample_data/top_800_mask/` 存在且包含 800 个 `*_masked.png` 文件

### Q2: KeyError: 'caption'
**原因**: top_800_info.csv 可能没有 caption 列
**解决**: 已在代码中添加默认处理 `item['caption'] = row['caption'] if 'caption' in row else ''`

### Q3: CUDA out of memory
**解决**:
1. 减小 `--batch_size` (如从16改为8)
2. 减小 `--map_dim` (如从768改为512)
3. 减少 `--num_mapping_layers`

### Q4: WandB login required
**解决**:
```bash
# 登录WandB（一次性）
wandb login
# 或者离线模式
export WANDB_MODE=offline
```

---

## 📊 与 pretraining 的对比

| 特性 | pretraining/ | 根目录 main.py |
|------|-------------|---------------|
| 任务 | CLIP投影层预训练 | 仇恨检测分类 |
| 训练内容 | 投影层 (1M) | 映射层+分类头 (~10M) |
| 损失函数 | 对比学习 | 交叉熵/BCE |
| 标签 | 无需标签 | 需要hateful标签 |
| 输出 | 优化的特征空间 | 分类预测 |

---

## ✅ 完整训练流程推荐

### 阶段1: 投影层预训练（可选）
```bash
cd pretraining
python main.py --freeze_strategy projection_only --max_epochs 20
```

### 阶段2: 快速测试
```bash
cd ..
powershell -ExecutionPolicy Bypass -File run_sample800_test.ps1
```

### 阶段3: 完整训练
```bash
powershell -ExecutionPolicy Bypass -File run_sample800_train.ps1
```

### 阶段4: 评估
检查 WandB 日志或 checkpoint 文件中的指标

---

## 🎓 总结

**Sample 800 训练的优势：**
- ✅ 数据量适中（800张），训练快速
- ✅ 包含完整的train/dev/test划分
- ✅ 支持masked和inpainted两种数据增强
- ✅ 可以验证模型架构和超参数
- ✅ 适合快速实验和原型开发

**适用场景：**
- 模型架构调试
- 超参数搜索
- 新想法快速验证
- 学习和理解代码

**如需更好效果：**
使用完整的10000张数据集（`--dataset original`）
