"""
WandB 登录脚本
用于快速登录 Weights & Biases
"""
import sys

def main():
    print("=" * 60)
    print("WandB (Weights & Biases) 登录")
    print("=" * 60)
    print()
    
    try:
        import wandb
    except ImportError:
        print("❌ 未安装 wandb，正在安装...")
        import subprocess
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'wandb'], check=True)
        import wandb
        print("✅ wandb 安装成功")
        print()
    
    print("步骤:")
    print("1. 访问 https://wandb.ai/authorize")
    print("2. 登录/注册后复制 API Key")
    print("3. 粘贴到下面")
    print()
    
    try:
        wandb.login()
        print("\n" + "=" * 60)
        print("✅ WandB 登录成功！")
        print("=" * 60)
        print("\n现在可以运行训练脚本了")
        print("例如: powershell -ExecutionPolicy Bypass -File run_sample800_train.ps1")
    except KeyboardInterrupt:
        print("\n\n❌ 用户取消登录")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 登录失败: {e}")
        print("\n你也可以手动运行: wandb login")
        sys.exit(1)

if __name__ == "__main__":
    main()
