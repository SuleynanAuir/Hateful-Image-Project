# 安装 CUDA 11.8 版本的 PyTorch (推荐，兼容性好)
# 文件大小约 2.6 GB，需要一定时间下载

Write-Host "正在安装 CUDA 版本的 PyTorch..." -ForegroundColor Green

# 使用 CUDA 11.8 (兼容 RTX 30/40 系列及更早的显卡)
C:/Users/Aiur/miniconda3/envs/hateful-image-ofa/python.exe -m pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

Write-Host "`n验证 CUDA 是否可用..." -ForegroundColor Green
C:/Users/Aiur/miniconda3/envs/hateful-image-ofa/python.exe -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

Write-Host "`n安装完成！" -ForegroundColor Green
