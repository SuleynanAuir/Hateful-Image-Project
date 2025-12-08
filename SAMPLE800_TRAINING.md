# Sample_800 训练指南

## ✅ WandB 已登录

你已经成功登录 WandB (suleynanaiur)，可以开始训练了！

---

## 🚀 快速开始

### 方式1: 使用脚本（推荐）

```powershell
# 完整训练（10 epochs，约15-20分钟）
powershell -ExecutionPolicy Bypass -File run_sample800_train.ps1

# 快速测试（2 epochs，10%数据，约3-5分钟）
powershell -ExecutionPolicy Bypass -File run_quick_test_main.ps1
```

### 方式2: 手动命令

```powershell
# 完整训练
C:/Users/Aiur/miniconda3/envs/hateful-image-ofa/python.exe main.py --dataset masked --labels original --freeze_image_encoder True --freeze_text_encoder True --fusion align --gpus 0 --batch_size 16 --lr 1e-4 --max_epochs 10 --log_every_n_steps 20

# 快速测试
C:/Users/Aiur/miniconda3/envs/hateful-image-ofa/python.exe main.py --dataset masked --labels original --freeze_image_encoder True --freeze_text_encoder True --fusion align --gpus 0 --batch_size 16 --lr 1e-4 --max_epochs 2 --limit_train_batches 0.1 --limit_val_batches 0.5 --log_every_n_steps 5
```

---

## 📊 数据集信息

### Sample_800 (Masked)
```
位置: data/sample_data/top_800_mask/
元数据: data/sample_data/top_800_info.csv

总计: 800 张图片
├── 训练集 (train): 659 张 (82.4%)
├── 验证集 (dev): 51 张 (6.4%)
└── 测试集 (test): 90 张 (11.2%)
```

---

## 🎯 模型架构

```
输入 Meme (图片 + 文本)
    │
    ├─→ Image Encoder (CLIP Vision) ❄️ 冻结
    │       ↓
    │   Image Map 🔥 训练
    │
    └─→ Text Encoder (CLIP Text) ❄️ 冻结
            ↓
        Text Map 🔥 训练
            │
            ↓
        Fusion (align) 🔥 训练
            ↓
        Classifier 🔥 训练
            ↓
        输出: Hateful (0/1)
```

**可训练参数**:
- Image Mapping Layer: 768 → 768
- Text Mapping Layer: 512/768 → 768  
- Fusion Layer: align (element-wise multiplication)
- Classification Head: 768 → 1 (binary)

**总参数**: ~5-10M (可训练) vs 151M (总计)

---

## 📈 训练配置

### 默认参数
```
--dataset masked              # 使用 masked 图片（800张）
--labels original             # 原始二分类标签
--freeze_image_encoder True   # 冻结图像编码器
--freeze_text_encoder True    # 冻结文本编码器
--fusion align                # CLIP默认融合方式
--gpus 0                      # 使用GPU 0 (RTX 4070)
--batch_size 16               # 批次大小
--lr 1e-4                     # 学习率
--max_epochs 10               # 训练轮数
```

### 监控指标
- **主要**: `val/auroc` (验证集 AUROC)
- **其他**: `val/acc`, `val/f1`, `val/precision`, `val/recall`

---

## 💾 输出文件

### Checkpoint
```
位置: checkpoints/
格式: {wandb_run_name}-epoch={XX}.ckpt
保存策略: 保存 val/auroc 最高的模型
```

### WandB 日志
```
在线查看: https://wandb.ai/suleynanaiur/meme-v2
本地日志: wandb/
```

可以在 WandB 网站查看：
- 训练/验证 Loss 曲线
- AUROC, F1, Accuracy 曲线
- 系统资源使用情况
- 超参数对比

---

## 🔧 如果需要重新登录

```powershell
# 运行登录脚本
C:/Users/Aiur/miniconda3/envs/hateful-image-ofa/python.exe setup_wandb.py

# 或者手动登录
wandb login
```

---

## 📝 参数调整建议

### 增加训练数据
```powershell
# 使用全部数据（去掉 limit 参数）
--max_epochs 10
# 不需要 --limit_train_batches 和 --limit_val_batches
```

### 调整模型容量
```powershell
# 更大的映射层
--map_dim 1024 --num_mapping_layers 2

# 更深的分类头
--num_pre_output_layers 2
```

### 不同融合策略
```powershell
# 拼接
--fusion concat

# 交叉注意力
--fusion cross

# 注意力机制
--fusion attention_m
```

### 加载预训练投影层
```powershell
--use_pretrained_map True
--local_pretrained_weights pretraining/checkpoints/model-epoch=19.ckpt
```

---

## ⚠️ 常见问题

### Q: CUDA out of memory
**解决**: 减小 batch_size
```powershell
--batch_size 8  # 或者 4
```

### Q: 训练太慢
**检查**: 
1. 确认使用 GPU: `--gpus 0`
2. 查看 GPU 利用率: `nvidia-smi`

### Q: 想查看训练进度
**方法**:
1. 终端实时输出
2. WandB 网页: https://wandb.ai/suleynanaiur/meme-v2

---

## 🎓 下一步

训练完成后：

1. **查看结果**
   - 检查 `checkpoints/` 中保存的模型
   - 在 WandB 查看训练曲线

2. **评估模型**
   - 训练脚本会自动在测试集上评估
   - 查看 test/auroc, test/f1 等指标

3. **使用其他数据集**
   ```powershell
   # 使用 inpainted 图片
   --dataset inpainted
   
   # 使用原始图片（10000张）
   --dataset original
   ```

4. **微调训练**
   ```powershell
   # 解冻部分层
   --freeze_image_encoder False --freeze_text_encoder True
   
   # 或全部解冻
   --freeze_image_encoder False --freeze_text_encoder False
   ```

---

**准备好了吗？运行训练吧！** 🚀

```powershell
powershell -ExecutionPolicy Bypass -File run_sample800_train.ps1
```
