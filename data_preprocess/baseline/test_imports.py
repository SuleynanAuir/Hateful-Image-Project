#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试OFA相关依赖是否可以导入"""

import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}\n")

# 测试基础库
try:
    import torch
    print(f"[OK] PyTorch: {torch.__version__}")
except ImportError as e:
    print(f"[FAIL] PyTorch import failed: {e}")

try:
    import fairseq
    print(f"[OK] Fairseq: {fairseq.__version__ if hasattr(fairseq, '__version__') else 'installed'}")
except ImportError as e:
    print(f"[FAIL] Fairseq import failed: {e}")

try:
    from fairseq import utils, tasks, checkpoint_utils
    print("[OK] Fairseq modules imported")
except ImportError as e:
    print(f"[FAIL] Fairseq modules import failed: {e}")

# 测试OFA特定模块
try:
    from utils.eval_utils import eval_step
    print("[OK] utils.eval_utils imported")
except ImportError as e:
    print(f"[FAIL] utils.eval_utils import failed: {e}")
    print("  需要安装OFA代码 (utils/, tasks/, models/ 目录)")

try:
    from tasks.mm_tasks.caption import CaptionTask
    print("[OK] tasks.mm_tasks.caption imported")
except ImportError as e:
    print(f"[FAIL] tasks.mm_tasks.caption import failed: {e}")
    print("  需要安装OFA代码")

try:
    from models.ofa import OFAModel
    print("[OK] models.ofa imported")
except ImportError as e:
    print(f"[FAIL] models.ofa import failed: {e}")
    print("  需要安装OFA代码")

# 测试其他依赖
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    from tqdm.auto import tqdm
    import psutil
    print("[OK] 其他依赖库导入成功")
except ImportError as e:
    print(f"[FAIL] 其他库导入失败: {e}")

# 检查文件路径
import os
print("\n检查文件路径:")
data_path = "../../data/all_data/info_fine_grained.csv"
if os.path.exists(data_path):
    print(f"[OK] 数据文件存在: {data_path}")
else:
    print(f"[FAIL] 数据文件不存在: {data_path}")

checkpoint_path = "checkpoints/caption.pt"
if os.path.exists(checkpoint_path):
    print(f"[OK] 模型权重存在: {checkpoint_path}")
else:
    print(f"[FAIL] 模型权重不存在: {checkpoint_path}")
    print("  需要从OFA官方GitHub下载模型权重")

img_folder = "../../data/hateful_memes_masked"
if os.path.exists(img_folder):
    print(f"[OK] 图像文件夹存在: {img_folder}")
else:
    print(f"[FAIL] 图像文件夹不存在: {img_folder}")
    # 检查是否有替代路径
    alt_img_folder = "../../data/img"
    if os.path.exists(alt_img_folder):
        print(f"  找到替代路径: {alt_img_folder}")

