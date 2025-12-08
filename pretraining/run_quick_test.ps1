# 快速测试脚本 - 使用10%数据快速验证投影层预训练

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  快速测试 - 投影层预训练" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "测试配置: 10% 训练数据, 3 epochs" -ForegroundColor Yellow
Write-Host "预计耗时: ~2分钟" -ForegroundColor Yellow
Write-Host ""

C:/Users/Aiur/miniconda3/envs/hateful-image-ofa/python.exe pretraining/main.py `
  --dataset masked `
  --image_pair text `
  --freeze_strategy projection_only `
  --gpus 0 `
  --batch_size 32 `
  --lr 5e-4 `
  --max_epochs 3 `
  --limit_train_batches 0.1 `
  --limit_val_batches 0.5 `
  --log_every_n_steps 2

Write-Host ""
Write-Host "测试完成！" -ForegroundColor Green
