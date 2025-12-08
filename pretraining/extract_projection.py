"""
从完整checkpoint中提取只训练过的投影层参数
"""
import torch
import argparse
import os

parser = argparse.ArgumentParser(description='Extract projection weights from checkpoint')
parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
parser.add_argument('--output', type=str, default='pretrained_projection.pth', help='Output file path')
args = parser.parse_args()

# 加载完整checkpoint
checkpoint = torch.load(args.checkpoint)

# 提取投影层参数
projection_weights = {
    k: v for k, v in checkpoint['state_dict'].items() 
    if 'projection' in k
}

print("提取的投影层参数:")
for name, param in projection_weights.items():
    print(f"  {name}: {param.shape}")

# 创建输出目录(如果不存在)
output_dir = os.path.dirname(args.output)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 保存为单独文件
torch.save(projection_weights, args.output)

print(f"\n总参数数: {sum(p.numel() for p in projection_weights.values()):,}")
print(f"已保存到: {args.output}")
