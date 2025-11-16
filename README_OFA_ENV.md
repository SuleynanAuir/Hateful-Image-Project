# OFA图像标题生成项目 - Conda环境配置指南

## 项目概述

本项目使用Meta的OFA (One-For-All)模型进行图像标题生成，用于Hateful Memes数据集的预处理工作。

## 环境依赖

### 核心库
- **Python**: 3.9 (推荐，fairseq对3.9支持较好)
- **PyTorch**: 1.12.0 (与fairseq兼容的版本)
- **Fairseq**: 从GitHub安装最新版本

### 数据处理库
- numpy, pandas
- matplotlib
- Pillow (PIL)
- tqdm, psutil

## 快速安装 (使用Miniconda)

### 方法1: 使用自动化脚本 (推荐)

**Windows:**
```bash
setup_ofa_env.bat
```

**Linux/Mac:**
```bash
chmod +x setup_ofa_env.sh
./setup_ofa_env.sh
```

### 方法2: 手动安装

1. **创建conda环境**
   ```bash
   conda env create -f environment_ofa.yml
   ```

2. **激活环境**
   ```bash
   conda activate hateful-image-ofa
   ```

3. **验证安装**
   ```bash
   python -c "import torch; import fairseq; print('安装成功！')"
   ```

## 重要：OFA模型代码安装

**注意**: 本项目代码引用了OFA特定的模块 (`utils.eval_utils`, `tasks.mm_tasks.caption`, `models.ofa`)，需要下载OFA的完整源码。

### 安装步骤

1. **克隆OFA仓库**
   ```bash
   git clone https://github.com/OFA-Sys/OFA.git
   cd OFA
   ```

2. **安装OFA**
   ```bash
   pip install -e .
   ```

3. **或者手动复制必要文件**
   将OFA仓库中的以下目录复制到本项目根目录：
   - `utils/` → 项目根目录的 `utils/`
   - `tasks/` → 项目根目录的 `tasks/`
   - `models/` → 项目根目录的 `models/`

4. **下载预训练模型权重**
   
   根据项目中的路径 `checkpoints/caption.pt`，需要下载OFA的caption任务预训练模型：
   
   ```bash
   # 创建checkpoints目录
   mkdir -p checkpoints
   
   # 下载caption模型（示例URL，请查看OFA官方文档获取最新链接）
   # 可以从 https://github.com/OFA-Sys/OFA 查看模型下载链接
   ```

## 验证环境

运行以下Python代码验证环境是否正确配置：

```python
import torch
import fairseq
from fairseq import utils, tasks, checkpoint_utils
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
```

## 使用环境

激活环境后运行notebook：
```bash
conda activate hateful-image-ofa
jupyter notebook data_preprocess/baseline/baseline_add_captions.ipynb
```

## 常见问题

### 1. fairseq安装失败
- 确保已安装构建依赖：`cython`, `hydra-core`, `omegaconf`
- 尝试安装特定版本：`pip install fairseq==0.12.2`

### 2. 找不到OFA模块
- 确保已克隆OFA仓库并正确安装
- 检查 `utils/`, `tasks/`, `models/` 目录是否存在

### 3. 模型权重文件缺失
- 从OFA官方GitHub下载预训练权重
- 确保权重文件路径与代码中的路径一致

## 系统要求

- **CPU**: 8核心 (根据检测结果)
- **内存**: 8GB+ (推荐16GB)
- **GPU**: 可选，但推荐使用GPU加速推理
- **存储**: 至少10GB可用空间（用于模型权重和数据集）

## 参考资源

- [OFA官方GitHub](https://github.com/OFA-Sys/OFA)
- [Fairseq官方文档](https://github.com/pytorch/fairseq)
- [PyTorch安装指南](https://pytorch.org/get-started/locally/)


