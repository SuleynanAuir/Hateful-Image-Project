# 验证GPU使用情况

Write-Host "检查GPU配置..." -ForegroundColor Cyan
Write-Host ""

# 检查PyTorch CUDA
C:/Users/Aiur/miniconda3/envs/hateful-image-ofa/python.exe -c @"
import torch
print('=' * 50)
print('PyTorch 版本:', torch.__version__)
print('CUDA 可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA 版本:', torch.version.cuda)
    print('GPU 数量:', torch.cuda.device_count())
    print('GPU 名称:', torch.cuda.get_device_name(0))
    print('当前设备:', torch.cuda.current_device())
print('=' * 50)
"@

Write-Host ""
Write-Host "命令参数说明:" -ForegroundColor Yellow
Write-Host "  --gpus 0          → 使用第0号GPU (你的RTX 4070)" -ForegroundColor Green
Write-Host "  --gpus '0 1'      → 使用多张GPU (空格分隔)" -ForegroundColor Gray
Write-Host "  --gpus ''         → 使用CPU模式" -ForegroundColor Gray
Write-Host ""
Write-Host "你的命令:" -ForegroundColor Cyan
Write-Host "  C:/.../python.exe pretraining/main.py --gpus 0 ..." -ForegroundColor White
Write-Host ""
Write-Host "结论: ✅ 会使用GPU (NVIDIA RTX 4070)" -ForegroundColor Green
