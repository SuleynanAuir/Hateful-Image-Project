# CLIP 投影层预训练脚本
# 只训练投影层（1M参数），在800张masked图片上预训练

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  CLIP 投影层预训练" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "数据集: 800张 masked 图片" -ForegroundColor Yellow
Write-Host "  - 训练集: 659张" -ForegroundColor Yellow
Write-Host "  - 验证集: 51张" -ForegroundColor Yellow
Write-Host "  - 测试集: 90张" -ForegroundColor Yellow
Write-Host ""
Write-Host "训练策略: projection_only" -ForegroundColor Green
Write-Host "  - 可训练参数: ~1M / 151M (0.7%)" -ForegroundColor Green
Write-Host "  - 冻结: Vision Encoder + Text Encoder" -ForegroundColor Green
Write-Host "  - 训练: Visual Projection + Text Projection" -ForegroundColor Green
Write-Host ""
Write-Host "训练参数:" -ForegroundColor Magenta
Write-Host "  - Batch Size: 32" -ForegroundColor Magenta
Write-Host "  - Learning Rate: 5e-4" -ForegroundColor Magenta
Write-Host "  - Max Epochs: 20" -ForegroundColor Magenta
Write-Host "  - GPU: NVIDIA RTX 4070" -ForegroundColor Magenta
Write-Host ""
Write-Host "预期效果:" -ForegroundColor Cyan
Write-Host "  - 训练速度: ~5-6 it/s" -ForegroundColor Cyan
Write-Host "  - 显存占用: ~4-6 GB" -ForegroundColor Cyan
Write-Host "  - Val Loss: 3.0 → 2.5" -ForegroundColor Cyan
Write-Host "  - 训练时长: ~30分钟" -ForegroundColor Cyan
Write-Host ""
Write-Host "开始训练..." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

C:/Users/Aiur/miniconda3/envs/hateful-image-ofa/python.exe pretraining/main.py `
  --dataset masked `
  --image_pair text `
  --freeze_strategy projection_only `
  --gpus 0 `
  --batch_size 32 `
  --lr 5e-4 `
  --max_epochs 20 `
  --log_every_n_steps 10

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  训练完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Checkpoint 保存位置: pretraining/checkpoints/" -ForegroundColor Yellow
Write-Host "最佳模型: model-epoch=XX.ckpt (val_loss最低)" -ForegroundColor Yellow
Write-Host ""
Write-Host "下一步: 在主训练脚本中加载预训练的投影层" -ForegroundColor Cyan
