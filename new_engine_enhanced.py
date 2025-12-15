import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from transformers import CLIPModel, AutoModelForCausalLM, AutoTokenizer
import os
import json
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("Warning: openai package not installed. Install with: pip install openai")


class SimpleGNNLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, edge_index):
        # x: [N, D], edge_index: [2, E]
        src, dst = edge_index
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])
        out = self.lin(self.norm(x + agg))
        out = F.relu(out)
        out = self.dropout(out)
        return x + out  # residual


class PCMModule(nn.Module):
    """Presupposed Context Module: h_PC = (W1 h_v + b1) ⊙ (W2 h_t + b2)."""
    def __init__(self, dim):
        super().__init__()
        self.lin_v = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(0.1), nn.LayerNorm(dim)
        )
        self.lin_t = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(0.1), nn.LayerNorm(dim)
        )

    def forward(self, h_v, h_t):
        # h_v: [B, D], h_t: [B, D]
        v_ctx = self.lin_v(h_v)
        t_ctx = self.lin_t(h_t)
        return v_ctx * t_ctx  # Hadamard product -> [B, D]


class FACTModule(nn.Module):
    """
    FACT = SPM (LLM-guided semantic prior)
         + CRM (Graph reasoning)
    """

    def __init__(
        self,
        dim,
        llm_dim=None,
        top_k=8,
        llm_name="gpt-5",
        use_api=True,
        api_key="sk-zk21884c49fc398427914f99fc30171ecb068f998dc6f05b",
        api_base= "https://api.zhizengzeng.com/v1",
    ):
        super().__init__()

        self.top_k = top_k
        self.llm_dim = llm_dim or dim

        # -------- LLM API --------
        self.api_client = openai.OpenAI(
            api_key=api_key,
            base_url=api_base,
        )

        # -------- Projection --------
        self.to_llm = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, self.llm_dim),
            nn.LayerNorm(self.llm_dim),
        )

        self.embed_resize = None

        # token update
        self.token_fuse = nn.Sequential(
            nn.Linear(self.llm_dim, self.llm_dim),
            nn.GELU(),
            nn.LayerNorm(self.llm_dim),
        )

        self.sp_pool = nn.Sequential(
            nn.Linear(self.llm_dim, self.llm_dim),
            nn.GELU(),
            nn.Linear(self.llm_dim, dim),
            nn.LayerNorm(dim),
        )

        # -------- CRM --------
        self.gnn1 = SimpleGNNLayer(dim)
        self.gnn2 = SimpleGNNLayer(dim)

    def build_spm(self, Hv, Ht, texts):
        """
        Hv: [B, P, D]
        Ht: [B, T, D]
        """
        B, P, _ = Hv.shape
        T = Ht.shape[1]
        device = Hv.device

        # ---- project tokens ----
        Hv_llm = self.to_llm(Hv)
        Ht_llm = self.to_llm(Ht)
        mm_tokens = torch.cat([Ht_llm, Hv_llm], dim=1)  # [B, N, d]
        N = T + P

        # ---- LLM prior ----
        emb = self.api_client.embeddings.create(
            input=texts,
            model="text-embedding-ada-002",
        )
        prior = torch.tensor(
            [e.embedding for e in emb.data],
            device=device,
            dtype=mm_tokens.dtype,
        )

        if prior.size(-1) != self.llm_dim:
            if self.embed_resize is None:
                self.embed_resize = nn.Linear(
                    prior.size(-1), self.llm_dim
                ).to(device)
            prior = self.embed_resize(prior)

        prior = F.normalize(prior, dim=-1)
        tokens_norm = F.normalize(mm_tokens, dim=-1)

        importance = torch.einsum("bd,bnd->bn", prior, tokens_norm)
        importance = torch.softmax(importance, dim=-1)

        # ---- PRIOR-CONDITIONED TOKEN UPDATE (关键) ----
        sp_tokens = mm_tokens + importance.unsqueeze(-1) * mm_tokens
        sp_tokens = self.token_fuse(sp_tokens)

        # ---- attention for graph ----
        scale = math.sqrt(self.llm_dim)
        A = torch.einsum("bnd,bmd->bnm", sp_tokens, sp_tokens) / scale
        A = A * importance.unsqueeze(1) * importance.unsqueeze(2)

        # ---- social perception vector ----
        h_sp = self.sp_pool(
            (sp_tokens * importance.unsqueeze(-1)).sum(dim=1)
        )

        return h_sp, sp_tokens, A

    def build_crm_edges(self, A, text_len, patch_len):
        B, N, _ = A.shape
        edges_all = []
        offset = 0

        for b in range(B):
            e = []

            # text chain
            for i in range(text_len - 1):
                e.append((offset + i, offset + i + 1))
                e.append((offset + i + 1, offset + i))

            # patch grid
            grid = int(math.sqrt(patch_len))
            base = offset + text_len

            for r in range(grid):
                for c in range(grid):
                    idx = base + r * grid + c
                    if r + 1 < grid:
                        e.append((idx, idx + grid))
                        e.append((idx + grid, idx))
                    if c + 1 < grid:
                        e.append((idx, idx + 1))
                        e.append((idx + 1, idx))

            # cross-modal
            A_tp = A[b, :text_len, text_len:]
            K = min(self.top_k, patch_len)
            topk = torch.topk(A_tp, K, dim=-1).indices

            for t in range(text_len):
                for k in range(K):
                    p = topk[t, k].item()
                    e.append((offset + t, base + p))
                    e.append((base + p, offset + t))

            edges_all.append(torch.tensor(e))
            offset += N

        edges_all = torch.cat(edges_all, dim=0)
        return edges_all.t().contiguous()

    def forward(self, Hv, Ht, texts):
        B, P, D = Hv.shape
        T = Ht.shape[1]

        h_sp, sp_tokens, A = self.build_spm(Hv, Ht, texts)

        # ---- CRM ----
        edge_index = self.build_crm_edges(A, T, P).to(Hv.device)
        x = sp_tokens.reshape(B * (T + P), self.llm_dim)

        x = self.gnn1(x, edge_index)
        x = self.gnn2(x, edge_index)

        x = x.reshape(B, T + P, self.llm_dim)
        h_cr = x.mean(dim=1)

        return h_sp, h_cr


class NewClassifier(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(vars(args))
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        # Load CLIP and optionally freeze encoders
        self.clip = CLIPModel.from_pretrained(args.clip_pretrained_model)
        self.image_encoder = copy.deepcopy(self.clip.vision_model)
        self.text_encoder = copy.deepcopy(self.clip.text_model)
        if args.freeze_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
        if args.freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
        # Projection to common dim with MLP instead of simple linear
        self.proj_dim = self.clip.projection_dim
        # infer hidden sizes
        d_img = getattr(self.image_encoder.config, 'hidden_size', self.clip.config.vision_config.hidden_size)
        d_txt = getattr(self.text_encoder.config, 'hidden_size', self.clip.config.text_config.hidden_size)
        # MLP projection for image features
        self.img_proj = nn.Sequential(
            nn.Linear(d_img, d_img),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d_img, self.proj_dim),
            nn.LayerNorm(self.proj_dim)
        )
        # MLP projection for text features
        self.txt_proj = nn.Sequential(
            nn.Linear(d_txt, d_txt),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d_txt, self.proj_dim),
            nn.LayerNorm(self.proj_dim)
        )
        # PCM and FACT (使用硬编码的 GPT-5 API)
        self.pcm = PCMModule(self.proj_dim)
        self.fact = FACTModule(self.proj_dim, llm_dim=self.proj_dim, llm_name='gpt-5')
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.proj_dim * 3, self.proj_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.proj_dim, 1)
        )
        # Auxiliary heads for per-modality losses to mimic main.py logging style
        self.img_head = nn.Linear(self.proj_dim, 1)
        self.txt_head = nn.Linear(self.proj_dim, 1)
        # Metrics
        self.auroc = torchmetrics.AUROC(task='binary')
        self.f1 = torchmetrics.F1Score(task='binary')
        self.acc = torchmetrics.Accuracy(task='binary')

    def encode_image(self, pixel_values):
        # pixel_values: [B, 3, H, W]
        out = self.image_encoder(pixel_values, output_hidden_states=True)
        # pooled embedding
        h_v_img = out.pooler_output  # [B, d_img]
        h_v = self.img_proj(h_v_img)  # [B, D]
        # patch-level embeddings from last hidden
        Hv_img = out.hidden_states[-1]  # [B, 1+P, d_img] (includes CLS)
        Hv_img = Hv_img[:, 1:, :]  # drop CLS
        Hv = self.img_proj(Hv_img)  # [B, P, D]
        return Hv, h_v

    def encode_text(self, input_ids, attention_mask):
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        h_t_txt = out.pooler_output  # [B, d_txt]
        h_t = self.txt_proj(h_t_txt)  # [B, D]
        Ht_txt = out.hidden_states[-1]  # [B, T, d_txt]
        Ht = self.txt_proj(Ht_txt)  # [B, T, D]
        return Ht, h_t

    def build_prompt(self, Hv, Ht):
        # Placeholder: use mean of Hv/Ht as prompt tokens
        return torch.cat([Hv.mean(dim=1, keepdim=True), Ht.mean(dim=1, keepdim=True)], dim=1)  # [B, 2, D]

    def forward(self, batch):
        pixel_values = batch['pixel_values'].squeeze(0)
        Hv, h_v = self.encode_image(pixel_values)       # [B,P,D], [B,D]
        Ht, h_t = self.encode_text(batch['input_ids'], batch['attention_mask'])  # [B,T,D], [B,D]
        # PCM
        h_pc = self.pcm(h_v, h_t)  # [B, D]
        # FACT (SPM + CRM)
        texts_for_llm = ["Are there false claims? " for _ in range(Hv.size(0))]
        h_sp, h_cr = self.fact(Hv, Ht, texts_for_llm)  # [B, D], [B, D]
        # Classifier: concat h_PC, h_SP, h_CR
        logits = self.classifier(torch.cat([h_pc, h_sp, h_cr], dim=-1)).squeeze(-1)  # [B]
        return logits
    

    def training_step(self, batch, batch_idx):
        # Forward for main classifier
        pixel_values = batch['pixel_values'].squeeze(0)
        Hv, h_v = self.encode_image(pixel_values)
        Ht, h_t = self.encode_text(batch['input_ids'], batch['attention_mask'])
        h_pc = self.pcm(h_v, h_t)
        texts_for_llm = ["Are there false claims? " for _ in range(Hv.size(0))]
        h_sp, h_cr = self.fact(Hv, Ht, texts_for_llm)
        main_logits = self.classifier(torch.cat([h_pc, h_sp, h_cr], dim=-1)).squeeze(-1)

        labels = batch['labels'].float()
        # Auxiliary modality-specific logits
        img_logits = self.img_head(h_v).squeeze(-1)
        txt_logits = self.txt_head(h_t).squeeze(-1)

        image_loss = F.binary_cross_entropy_with_logits(img_logits, labels)
        text_loss = F.binary_cross_entropy_with_logits(txt_logits, labels)
        cls_loss = F.binary_cross_entropy_with_logits(main_logits, labels)
        total_loss = cls_loss + 0.5 * (image_loss + text_loss)

        # Metrics on main logits
        preds = (torch.sigmoid(main_logits) > 0.5).long()
        valid_mask = (labels >= 0) & (labels <= 1)
        if valid_mask.sum() > 0:
            valid_logits = torch.sigmoid(main_logits[valid_mask])
            valid_preds = preds[valid_mask]
            valid_labels = labels[valid_mask].int()
            self.log('train/auroc', self.auroc(valid_logits, valid_labels), prog_bar=True, on_step=True, on_epoch=False)
            self.log('train/accuracy', self.acc(valid_preds, valid_labels), prog_bar=True, on_step=True, on_epoch=False)
            self.log('train/f1', self.f1(valid_preds, valid_labels), prog_bar=False, on_step=True, on_epoch=False)

        # Log losses following original main style
        self.log('train/total_loss', total_loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log('train/image_loss', image_loss, prog_bar=False, on_step=True, on_epoch=False)
        self.log('train/text_loss', text_loss, prog_bar=False, on_step=True, on_epoch=False)
        self.log('train/loss', cls_loss, prog_bar=True, on_step=True, on_epoch=False)
        return total_loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        labels = batch['labels'].float()
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        preds = (torch.sigmoid(logits) > 0.5).long()
        acc = (preds == labels.long()).float().mean()
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/acc', acc, prog_bar=True)
        # Filter out samples with label -1 for metrics (torchmetrics only accepts [0, 1])
        valid_mask = (labels >= 0) & (labels <= 1)
        if valid_mask.sum() > 0:
            valid_logits = torch.sigmoid(logits[valid_mask])
            valid_preds = preds[valid_mask]
            valid_labels = labels[valid_mask].int()
            self.log('val/auroc', self.auroc(valid_logits, valid_labels), prog_bar=True)
            self.log('val/f1', self.f1(valid_preds, valid_labels), prog_bar=True)
        return {'loss': loss, 'acc': acc}

    # def test_step(self, batch, batch_idx):
    #     logits = self(batch)
    #     labels = batch['labels'].float()
    #     loss = F.binary_cross_entropy_with_logits(logits, labels)
    #     preds = (torch.sigmoid(logits) > 0.5).long()
    #     acc = (preds == labels.long()).float().mean()
    #     self.log('test/loss', loss, prog_bar=True)
    #     self.log('test/acc', acc, prog_bar=True)
    #     self.log('test/auroc', self.auroc(torch.sigmoid(logits), labels.int()), prog_bar=True)
    #     self.log('test/f1', self.f1(preds, labels.int()), prog_bar=True)
    #     return {'loss': loss, 'acc': acc}
    def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
        logits = self(batch)
        labels = batch['labels'].float()
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        preds = (torch.sigmoid(logits) > 0.5).long()
        acc = (preds == labels.long()).float().mean()
        prefix = f'test{dataloader_idx}'
        self.log(f'{prefix}/loss', loss, prog_bar=True)
        self.log(f'{prefix}/acc', acc, prog_bar=True)
        # Filter out samples with label -1 for metrics (torchmetrics only accepts [0, 1])
        valid_mask = (labels >= 0) & (labels <= 1)
        if valid_mask.sum() > 0:
            valid_logits = torch.sigmoid(logits[valid_mask])
            valid_preds = preds[valid_mask]
            valid_labels = labels[valid_mask].int()
            self.log(f'{prefix}/auroc', self.auroc(valid_logits, valid_labels), prog_bar=True)
            self.log(f'{prefix}/f1', self.f1(valid_preds, valid_labels), prog_bar=True)
        return {'loss': loss, 'acc': acc}

    def configure_optimizers(self):
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
