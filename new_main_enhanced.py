import argparse
import warnings
# Filter out the repetitive pkg_resources deprecation warnings from lightning_fabric
warnings.filterwarnings('ignore', message='.*pkg_resources is deprecated.*')

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets import load_dataset, CustomCollator
from new_engine_enhanced import NewClassifier


def get_arg_parser():
    p = argparse.ArgumentParser(description='New pipeline for Hateful Memes Detection')
    p.add_argument('--dataset', default='inpainted', choices=['original','masked','inpainted'])
    p.add_argument('--labels', default='original')
    p.add_argument('--image_size', type=int, default=224)
    p.add_argument('--clip_pretrained_model', type=str, default='openai/clip-vit-base-patch32')
    p.add_argument('--caption_mode', type=str, default='none', choices=['none','replace_text','concat_with_text'])
    p.add_argument('--multilingual_tokenizer_path', type=str, default='none')
    p.add_argument('--freeze_image_encoder', type=bool, default=True)
    p.add_argument('--freeze_text_encoder', type=bool, default=True)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--gpus', default='0')
    p.add_argument('--max_epochs', type=int, default=20)
    p.add_argument('--log_every_n_steps', type=int, default=10)
    p.add_argument('--limit_train_batches', type=float, default=1.0)
    p.add_argument('--limit_val_batches', type=float, default=1.0)
    p.add_argument('--use_pretrained_map', type=bool, default=False, help='Use pretrained projection weights')
    p.add_argument('--local_pretrained_weights', type=str, default=None, help='Path to pretrained weights file')
    p.add_argument('--wandb_mode', type=str, default='offline', choices=['offline','online','disabled'], help='WandB logging mode')
    p.add_argument('--wandb_timeout', type=int, default=180, help='WandB init timeout seconds')
    p.add_argument('--wandb_base_url', type=str, default='https://api.wandb.ai', help='Base URL for WandB API')
    return p


def main(args):
    # datasets: train/eval/test
    dataset_train = load_dataset(args=args, split='train')
    dataset_val = load_dataset(args=args, split='eval')
    dataset_test = load_dataset(args=args, split='test')
    print('Number of training examples:', len(dataset_train))
    print('Number of eval examples:', len(dataset_val))
    print('Number of test examples:', len(dataset_test))
    
    # Validate limit_*_batches to ensure at least 1 batch will be processed
    num_val_batches = len(dataset_val) // args.batch_size + (1 if len(dataset_val) % args.batch_size else 0)
    if args.limit_val_batches < 1.0 and args.limit_val_batches * num_val_batches < 1:
        min_limit = 1.0 / num_val_batches
        print(f'Warning: limit_val_batches={args.limit_val_batches} is too small for {num_val_batches} batches.')
        print(f'Adjusting to minimum valid value: {min_limit:.3f}')
        args.limit_val_batches = min_limit

    collator = CustomCollator(args, dataset_train.fine_grained_labels, multilingual_tokenizer_path=args.multilingual_tokenizer_path)
    num_workers = min(args.batch_size, 16)
    dl_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, collate_fn=collator)
    dl_val = DataLoader(dataset_val, batch_size=args.batch_size, num_workers=num_workers, collate_fn=collator)
    dl_test = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=num_workers, collate_fn=collator)

    seed_everything(42, workers=True)
    model = NewClassifier(args)
    
    # Load pretrained weights if specified
    if args.use_pretrained_map and args.local_pretrained_weights:
        import torch
        print(f"Loading pretrained weights from {args.local_pretrained_weights}")
        try:
            pretrained_dict = torch.load(args.local_pretrained_weights, map_location='cpu')
            model_dict = model.state_dict()
            # Filter out keys that match img_proj and txt_proj
            pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                             if k in model_dict and ('img_proj' in k or 'txt_proj' in k)}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
            print(f"✓ Loaded {len(pretrained_dict)} pretrained parameters")
        except Exception as e:
            print(f"Warning: Failed to load pretrained weights: {e}")

    # Simplified WandB initialization: single attempt, fallback offline, no retries
    import wandb, time, os
    os.environ.setdefault('WANDB_BASE_URL', args.wandb_base_url)
    run = None
    wandb_logger = None
    if args.wandb_mode == 'online':
        try:
            settings = wandb.Settings(init_timeout=args.wandb_timeout)
            run = wandb.init(project='meme-v2', config=vars(args), mode='online', settings=settings)
        except Exception as e:
            print(f"WandB online init failed: {e}")
            print("Switching to offline mode (no retry).")
            args.wandb_mode = 'offline'
    if run is None and args.wandb_mode == 'offline':
        try:
            settings = wandb.Settings(init_timeout=min(args.wandb_timeout, 300))
            run = wandb.init(project='meme-new-pipeline', config=vars(args), mode='offline', settings=settings)
        except Exception as e:
            print(f"WandB offline init failed: {e}; disabling WandB logging.")
            args.wandb_mode = 'disabled'
    run_name = None
    if run is not None:
        if run.name is None:
            prefix = getattr(args, 'dataset', 'run')
            rid = (getattr(run, 'id', '') or '')[:8] or str(int(time.time()))
            run.name = f"{prefix}-{rid}"
        run_name = run.name
        wandb_logger = WandbLogger(experiment=run)
    else:
        print('WandB logger not active; proceeding without WandB.')
        # generate a fallback run_name for checkpointing
        prefix = getattr(args, 'dataset', 'run')
        run_name = f"{prefix}-{int(time.time())}"
        wandb_logger = None
    ckpt = ModelCheckpoint(dirpath='checkpoints_new', filename=run_name+'-{epoch:02d}',
                           monitor='val/acc', mode='max', save_top_k=1)
    
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=args.max_epochs,
        # 可以添加这一行让控制台输出更紧凑，避免刷屏 (可选)
        # enable_progress_bar=True, 
    )

    # 【核心修改点】: 将 val_dataloaders 设置为一个列表 [dl_val, dl_test]
    # 这样 validation_step 会依次处理这两个数据集
    print("Starting training (validating on both Eval and Test sets)...")
    trainer.fit(model, train_dataloaders=dl_train, val_dataloaders=[dl_val, dl_test])
    # 训练结束后，依然可以使用 best model 再跑一次最终测试
    print("Running final test with best checkpoint...")
    trainer.test(ckpt_path='best', dataloaders=[dl_val, dl_test])


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
