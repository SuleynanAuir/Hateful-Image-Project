# 运行 CLIP 预训练脚本
# 使用 masked 数据集，text 文本对，小批量快速测试

Write-Host "=== 开始训练 ===" -ForegroundColor Green
Write-Host "数据集: masked (top_800_mask)" -ForegroundColor Cyan
Write-Host "Batch size: 8" -ForegroundColor Cyan
Write-Host "Max epochs: 2" -ForegroundColor Cyan
Write-Host "训练样本: 1% (约85个)" -ForegroundColor Cyan
Write-Host "验证样本: 5% (约25个)" -ForegroundColor Cyan
Write-Host ""

# 禁用 WandB 在线同步（可选）
$env:WANDB_MODE = "offline"

# 运行训练
C:/Users/Aiur/miniconda3/envs/hateful-image-ofa/python.exe main.py `
  --dataset masked `
  --image_pair text `
  --gpus 0 `
  --batch_size 8 `
  --lr 1e-4 `
  --max_epochs 2 `
  --limit_train_batches 0.01 `
  --limit_val_batches 0.05 `
  --log_every_n_steps 5 `
  --val_check_interval 1.0

Write-Host "`n=== 训练完成 ===" -ForegroundColor Green
Write-Host "Checkpoint 保存在: pretraining/checkpoints/" -ForegroundColor Cyan
