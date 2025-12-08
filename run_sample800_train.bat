@echo off
REM 使用 Sample_800 数据集训练 Hateful Memes 分类模型
REM 训练集: 659张, 验证集: 51张, 测试集: 90张

echo ========================================
echo   Hateful Memes 分类训练 (Sample 800)
echo ========================================
echo.
echo 数据集: 800张 masked 图片 (top_800_mask)
echo   - 训练集: 659张
echo   - 验证集: 51张
echo   - 测试集: 90张
echo.
echo 模型配置:
echo   - Image Encoder: CLIP Vision (冻结)
echo   - Text Encoder: CLIP Text (冻结)
echo   - 训练: Mapping + Fusion + Classification Head
echo.
echo 训练参数:
echo   - Batch Size: 16
echo   - Learning Rate: 1e-4
echo   - Max Epochs: 10
echo   - Fusion: align (CLIP default)
echo.
echo 开始训练...
echo ========================================
echo.

C:\Users\Aiur\miniconda3\envs\hateful-image-ofa\python.exe main.py ^
  --dataset masked ^
  --labels original ^
  --freeze_image_encoder True ^
  --freeze_text_encoder True ^
  --fusion align ^
  --gpus 0 ^
  --batch_size 16 ^
  --lr 1e-4 ^
  --max_epochs 10 ^
  --log_every_n_steps 20 ^
  --num_mapping_layers 1 ^
  --map_dim 768 ^
  --num_pre_output_layers 1

echo.
echo ========================================
echo   训练完成！
echo ========================================
echo.
echo Checkpoint 保存位置: checkpoints\
echo 监控指标: val/auroc (验证集 AUROC)
echo.
pause
