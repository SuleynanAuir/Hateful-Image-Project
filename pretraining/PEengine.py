import pytorch_lightning as pl
import torch
from types import SimpleNamespace

from core.vision_encoder.factory import create_model_and_transforms, get_tokenizer


class PEAsCLIP(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.lr = args.lr
        self.weight_decay = args.weight_decay


        self.model, _, _ = create_model_and_transforms(
            args.pe_model_name,
            pretrained=True
        )
        self.tokenizer = get_tokenizer(args.pe_model_name)

        # CLIP 风格的温度参数（logit_scale）
        if hasattr(self.model, "logit_scale"):
            self.logit_scale = self.model.logit_scale
        else:
            self.logit_scale = torch.nn.Parameter(torch.tensor(4.6052))  # default exp(4.6052)=100


    def forward(self, batch):
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # 1. 编码图像 & 文本
        image_embeds = self.model.encode_image(pixel_values)
        text_embeds = self.model.encode_text(input_ids, attention_mask=attention_mask)

        # 2. L2 归一化（与 CLIP 完全一致）
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # 3. logits_per_image, logits_per_text（完全仿 CLIP）
        scale = self.logit_scale.exp()
        logits_per_image = scale * image_embeds @ text_embeds.t()
        logits_per_text = logits_per_image.t()

        # 4. 如果提供 labels，则计算 cross-modal 对比损失（等同 CLIP 的 return_loss=True)
        loss = None
        if "labels" in batch:
            labels = batch["labels"]
            loss_i2t = torch.nn.functional.cross_entropy(logits_per_image, labels)
            loss_t2i = torch.nn.functional.cross_entropy(logits_per_text, labels)
            loss = (loss_i2t + loss_t2i) / 2

        # 5. 返回“完全 CLIP 格式”的对象
        return SimpleNamespace(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            image_embeds=image_embeds,
            text_embeds=text_embeds
        )


    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = output.loss
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = output.loss
        self.log("val/loss", loss)
        return loss

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        return optimizer


def create_model(args):
    return PEAsCLIP(args)
