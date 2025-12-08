import pytorch_lightning as pl
import torch
from transformers import CLIPModel



class HatefulMemesCLIP(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.freeze_strategy = args.freeze_strategy
        self.clip = CLIPModel.from_pretrained(args.clip_pretrained_model)
        
        # 应用冻结策略
        self._apply_freeze_strategy()

    def _apply_freeze_strategy(self):
        """
        应用不同的冻结策略：
        - 'none': 不冻结任何层（全模型训练）
        - 'projection_only': 只训练投影层（visual_projection, text_projection）
        - 'vision_encoder': 冻结视觉编码器，训练文本编码器和投影层
        - 'text_encoder': 冻结文本编码器，训练视觉编码器和投影层
        - 'encoders': 冻结两个编码器，只训练投影层
        """
        if self.freeze_strategy == 'none':
            # 全模型训练，不冻结
            return
        
        # 首先冻结所有参数
        for param in self.clip.parameters():
            param.requires_grad = False
        
        if self.freeze_strategy == 'projection_only':
            # 只训练投影层
            if hasattr(self.clip, 'visual_projection'):
                for param in self.clip.visual_projection.parameters():
                    param.requires_grad = True
            if hasattr(self.clip, 'text_projection'):
                for param in self.clip.text_projection.parameters():
                    param.requires_grad = True
            print("🔒 冻结策略: 只训练投影层 (visual_projection + text_projection)")
        
        elif self.freeze_strategy == 'vision_encoder':
            # 冻结视觉编码器
            for param in self.clip.text_model.parameters():
                param.requires_grad = True
            if hasattr(self.clip, 'visual_projection'):
                for param in self.clip.visual_projection.parameters():
                    param.requires_grad = True
            if hasattr(self.clip, 'text_projection'):
                for param in self.clip.text_projection.parameters():
                    param.requires_grad = True
            print("🔒 冻结策略: 冻结视觉编码器，训练文本编码器和投影层")
        
        elif self.freeze_strategy == 'text_encoder':
            # 冻结文本编码器
            for param in self.clip.vision_model.parameters():
                param.requires_grad = True
            if hasattr(self.clip, 'visual_projection'):
                for param in self.clip.visual_projection.parameters():
                    param.requires_grad = True
            if hasattr(self.clip, 'text_projection'):
                for param in self.clip.text_projection.parameters():
                    param.requires_grad = True
            print("🔒 冻结策略: 冻结文本编码器，训练视觉编码器和投影层")
        
        elif self.freeze_strategy == 'encoders':
            # 冻结编码器，只训练投影层
            if hasattr(self.clip, 'visual_projection'):
                for param in self.clip.visual_projection.parameters():
                    param.requires_grad = True
            if hasattr(self.clip, 'text_projection'):
                for param in self.clip.text_projection.parameters():
                    param.requires_grad = True
            print("🔒 冻结策略: 冻结两个编码器，只训练投影层")
        
        # 打印可训练参数统计
        trainable_params = sum(p.numel() for p in self.clip.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.clip.parameters())
        print(f"📊 可训练参数: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    def forward(self, batch):
        output = self.clip(pixel_values=batch['pixel_values'][0], input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], return_loss=True, return_dict=True)

        return output
        
    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = output.loss
        self.log('train/loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = output.loss
        self.log('val/loss', loss)

        return loss

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if p.requires_grad]}
            ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

        return optimizer



def create_model(args):
    model = HatefulMemesCLIP(args=args)

    return model