import pytorch_lightning as pl
import torch
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms

class HatefulMemesCLIP(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        
        self.model = pe.CLIP.from_config(args.clip_config, pretrained=True)

        self.context_length = self.model.context_length

    def forward(self, batch):
        image_features, text_features, logit_scale = self.model(
            batch['pixel_values'][0],  
            batch['input_ids']        
        )
        
        output = type('Object', (object,), {})()
        output.loss = self.compute_contrastive_loss(image_features, text_features, logit_scale)
        
        return output
    
    def compute_contrastive_loss(self, image_features, text_features, logit_scale):
        """计算CLIP对比损失"""
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T

        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=self.device)

        loss_i = torch.nn.functional.cross_entropy(logits_per_image, labels)
        loss_t = torch.nn.functional.cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2
        
        return loss

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