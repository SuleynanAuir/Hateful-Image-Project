# 运行 baseline_add_captions.ipynb 的准备工作

## 当前状态

✅ Conda环境已创建: `hateful-image-ofa`
✅ Fairseq 0.12.2 已安装
✅ Jupyter 已安装并配置kernel
⚠️ PyTorch DLL加载问题需要修复
❌ OFA代码未安装 (需要 utils/, tasks/, models/ 目录)
❌ 模型权重文件缺失 (需要 checkpoints/caption.pt)
❌ 图像文件夹路径需要确认

## 需要完成的步骤

### 1. 修复PyTorch DLL问题 (如果遇到)

如果遇到 `OSError: [WinError 182]` 错误，尝试：

```bash
conda activate hateful-image-ofa
pip uninstall torch torchvision torchaudio -y
pip install torch==1.12.0+cpu torchvision==0.13.0+cpu torchaudio==0.12.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

或者安装CPU版本：
```bash
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cpuonly -c pytorch
```

### 2. 安装OFA代码

从OFA官方GitHub克隆并安装：

```bash
cd C:\Users\Aiur\Hateful-Image-Project
git clone https://github.com/OFA-Sys/OFA.git
cd OFA
pip install -e .
cd ..
```

这将创建必要的 `utils/`, `tasks/`, `models/` 目录。

### 3. 下载模型权重

需要从OFA官方下载caption任务预训练模型：

1. 创建checkpoints目录：
   ```bash
   mkdir checkpoints
   ```

2. 下载caption.pt模型权重到 `checkpoints/caption.pt`
   - 参考: https://github.com/OFA-Sys/OFA
   - 或使用提供的下载链接

### 4. 确认数据路径

Notebook中使用以下路径：
- 数据文件: `../../data/hateful_memes/info_fine_grained.csv`
  - 实际路径应该是: `data/all_data/info_fine_grained.csv` ✓ 已存在
- 图像文件夹: `../../data/hateful_memes_masked`
  - 如果不存在，可能需要修改为: `../../data/img`

### 5. 运行Notebook

有两种方式：

**方式1: 使用Jupyter Notebook (推荐)**
```bash
conda activate hateful-image-ofa
cd C:\Users\Aiur\Hateful-Image-Project\data_preprocess\baseline
jupyter notebook baseline_add_captions.ipynb
```

**方式2: 使用JupyterLab**
```bash
conda activate hateful-image-ofa
cd C:\Users\Aiur\Hateful-Image-Project
jupyter lab
```

在打开的notebook中，确保kernel选择为 "Python (hateful-image-ofa)"

## 如果遇到问题

1. **PyTorch导入错误**: 参考步骤1重新安装PyTorch
2. **OFA模块找不到**: 确保已执行步骤2安装OFA代码
3. **模型权重缺失**: 确保已执行步骤3下载模型权重
4. **图像路径错误**: 检查并修改notebook中的图像文件夹路径

## 快速检查脚本

运行 `test_imports.py` 检查所有依赖：
```bash
cd C:\Users\Aiur\Hateful-Image-Project\data_preprocess\baseline
conda activate hateful-image-ofa
python test_imports.py
```

