# 快速测试 - 使用 10% 数据验证代码是否正常运行

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  快速测试 - Sample 800 训练" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "测试配置: 10% 训练数据, 2 epochs" -ForegroundColor Yellow
Write-Host "预计耗时: ~3-5 分钟" -ForegroundColor Yellow
Write-Host ""

C:/Users/Aiur/miniconda3/envs/hateful-image-ofa/python.exe main.py `
  --dataset masked `
  --labels original `
  --freeze_image_encoder True `
  --freeze_text_encoder True `
  --fusion align `
  --gpus 0 `
  --batch_size 16 `
  --lr 1e-4 `
  --max_epochs 2 `
  --limit_train_batches 0.1 `
  --limit_val_batches 0.5 `
  --log_every_n_steps 5

Write-Host ""
Write-Host "测试完成！如果没有错误，可以运行完整训练。" -ForegroundColor Green
