from __future__ import annotations

import math
import os

import torch
from torch import nn
from torch.nn import functional as F

from .config import Config


class FuzzyLogic(nn.Module):
    def __init__(self, norm: str = 'godel'):
        super().__init__()
        if norm not in {'godel', 'product', 'lukasiewicz'}:
            raise ValueError('Unknown t-norm')
        self.norm = norm
        self.log_or_temperature = nn.Parameter(torch.tensor(-2.5))
        self.log_and_temperature = nn.Parameter(torch.tensor(-2.5))

    def disjunction(self, x: torch.Tensor, membership: torch.Tensor) -> torch.Tensor:
        if self.norm == 'godel':
            temperature = self.log_or_temperature.exp().clamp(0.005, 1)
            weights = membership / membership.sum(-1, keepdim=True).clamp_min(1e-8)
            return (temperature * torch.logsumexp(x / temperature + weights.clamp_min(1e-30).log(), -1)).clamp(0, 1)
        selected = x * membership
        if self.norm == 'product':
            return 1 - (1 - selected).prod(-1)
        return selected.sum(-1).clamp(0, 1)

    def conjunction(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        mask = torch.ones_like(x) if mask is None else mask.to(x.dtype)
        if self.norm == 'godel':
            temperature = self.log_and_temperature.exp().clamp(0.005, 1)
            logits = (-x / temperature).masked_fill(mask == 0, -torch.inf)
            return (-temperature * (torch.logsumexp(logits, -1) - mask.sum(-1).clamp_min(1).log())).clamp(0, 1)
        included = 1 - mask * (1 - x)
        if self.norm == 'product':
            return included.prod(-1)
        return (included.sum(-1) - included.shape[-1] + 1).clamp(0, 1)

    def forward(self, activations: torch.Tensor, membership: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        groups = self.disjunction(activations[:, None, None, :], membership[None, :, :, :])
        return self.conjunction(groups), groups


class TextEncoder(nn.Module):
    def __init__(self, cfg: Config, restore_directory=None):
        super().__init__()
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        location = str(restore_directory or cfg.encoder)
        self.tokenizer = AutoTokenizer.from_pretrained(location, revision=None if restore_directory else cfg.encoder_revision,
                                                       local_files_only=True, trust_remote_code=False)
        if cfg.freeze_encoder:
            self.encoder = AutoModel.from_pretrained(cfg.encoder, revision=cfg.encoder_revision,
                                                     local_files_only=True, trust_remote_code=False)
            self.encoder.requires_grad_(False)
            self.encoder.eval()
        elif restore_directory:
            self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(location, local_files_only=True))
        else:
            self.encoder = AutoModel.from_pretrained(location, local_files_only=True, trust_remote_code=False)
        if hasattr(self.encoder, 'gradient_checkpointing_enable'):
            self.encoder.gradient_checkpointing_enable()
        self.cfg = cfg
        cfg.encoder_revision = cfg.encoder_revision or getattr(self.encoder.config, '_commit_hash', None)
        width = self.encoder.config.hidden_size
        self.pool_attention = nn.Linear(width, 1)
        self.projection = nn.Linear(width, cfg.dimension)
        self.feature_store = None

    def train(self, mode: bool = True):
        super().train(mode)
        if self.cfg.freeze_encoder:
            self.encoder.eval()
        return self

    def enable_feature_cache(self):
        if self.cfg.freeze_encoder and self.cfg.pooling != 'attention':
            from .features import FeatureStore
            self.feature_store = FeatureStore(self.cfg, self.cfg.encoder_revision or str(self.cfg.encoder))

    def forward(self, texts: list[str], statute: bool = False) -> torch.Tensor:
        if self.feature_store is not None:
            cached = [self.feature_store.get(text, statute) for text in texts]
            missing = [i for i, vector in enumerate(cached) if vector is None]
            if missing:
                vectors = self.pooled([texts[i] for i in missing], statute).detach().cpu().numpy()
                for i, vector in zip(missing, vectors):
                    self.feature_store.put(texts[i], statute, vector)
                    cached[i] = vector
                self.feature_store.commit()
            import numpy as np
            features = torch.from_numpy(np.stack(cached)).to(self.projection.weight.device)
        else:
            features = self.pooled(texts, statute)
        return F.normalize(self.projection(features).float(), dim=-1)

    def pooled(self, texts: list[str], statute: bool = False) -> torch.Tensor:
        device = self.projection.weight.device
        length = self.cfg.statute_length if statute else self.cfg.max_length
        chunked = not statute and self.cfg.truncation == 'chunked'
        batch = self.tokenizer(texts, truncation=True, max_length=length, padding=True,
                               return_overflowing_tokens=chunked, stride=64 if chunked else 0,
                               return_tensors='pt')
        owners = batch.pop('overflow_to_sample_mapping', None)
        if chunked:
            counts: dict[int, int] = {}
            keep = []
            for i, owner in enumerate(owners.tolist()):
                counts[owner] = counts.get(owner, 0) + 1
                if counts[owner] <= self.cfg.max_chunks:
                    keep.append(i)
            batch = {key: value[keep] for key, value in batch.items()}
            owners = owners[keep]
        outputs = []
        size = self.cfg.feature_batch_size if self.cfg.freeze_encoder else self.cfg.batch_size
        for start in range(0, len(batch['input_ids']), size):
            part = {key: value[start:start + size].to(device) for key, value in batch.items()}
            if self.cfg.freeze_encoder:
                with torch.no_grad():
                    hidden = self.encoder(**part).last_hidden_state
            else:
                hidden = self.encoder(**part).last_hidden_state
            mask = part['attention_mask'].unsqueeze(-1)
            if self.cfg.pooling == 'cls':
                pooled = hidden[:, 0]
            elif self.cfg.pooling == 'attention':
                weights = self.pool_attention(hidden).masked_fill(mask == 0, -1e4).softmax(1)
                pooled = (hidden * weights).sum(1)
            else:
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            outputs.append(pooled)
        encoded = torch.cat(outputs)
        if chunked:
            encoded = torch.stack([encoded[owners.to(device) == i].mean(0) for i in range(len(texts))])
        return encoded.float()


class LexForm(nn.Module):
    def __init__(self, cfg: Config, sections: int, text_encoder: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.text_encoder = text_encoder
        self.bank = nn.Parameter(torch.randn(cfg.bank_size, cfg.dimension) / math.sqrt(cfg.dimension))
        self.form_queries = nn.Linear(cfg.dimension, cfg.groups * cfg.dimension)
        self.fact_projection = nn.Linear(cfg.dimension, cfg.dimension)
        self.activation_scale = nn.Parameter(torch.tensor(3.0))
        self.activation_bias = nn.Parameter(torch.zeros(cfg.bank_size))
        self.logic = FuzzyLogic(cfg.norm)
        self.flat_head = nn.Linear(cfg.dimension, sections) if cfg.variant == 'flat' else None
        if cfg.variant == 'per_section':
            self.private_bank = nn.Parameter(torch.randn(sections, cfg.bank_size, cfg.dimension) / math.sqrt(cfg.dimension))

    def induce(self, section_vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        queries = self.form_queries(section_vectors).view(-1, self.cfg.groups, self.cfg.dimension)
        bank = F.normalize(self.bank, dim=-1)
        if self.cfg.variant == 'per_section':
            logits = torch.einsum('skd,sed->ske', F.normalize(queries, dim=-1), F.normalize(self.private_bank, dim=-1)) * 8
        else:
            logits = torch.einsum('skd,ed->ske', F.normalize(queries, dim=-1), bank) * 8
        probabilities = logits.softmax(-1)
        count = min(4, self.cfg.bank_size)
        sparse = torch.zeros_like(probabilities).scatter_(-1, probabilities.topk(count, -1).indices, 1)
        membership = probabilities * sparse
        membership = membership / membership.amax(-1, keepdim=True).clamp_min(1e-8)
        return membership, probabilities

    def forward(self, facts: list[str], statutes: list[str], fixed_forms=None, section_vectors=None) -> dict:
        fact_vectors = self.text_encoder(facts)
        if self.flat_head is not None:
            scores = self.flat_head(fact_vectors).sigmoid()
            return {'scores': scores, 'groups': scores.unsqueeze(-1)}
        if section_vectors is None:
            section_vectors = self.text_encoder(statutes, statute=True)
        membership, probabilities = self.induce(section_vectors)
        if fixed_forms is not None:
            membership = fixed_forms.to(fact_vectors.device)
        projected = F.normalize(self.fact_projection(fact_vectors), dim=-1)
        if self.cfg.variant == 'per_section':
            activations = (torch.einsum('bd,sed->bse', projected, F.normalize(self.private_bank, dim=-1))
                           * self.activation_scale.exp().clamp(max=50) + self.activation_bias).sigmoid()
            group_scores = self.logic.disjunction(activations[:, :, None, :], membership[None])
            scores = self.logic.conjunction(group_scores)
        else:
            activations = (projected @ F.normalize(self.bank, dim=-1).T
                           * self.activation_scale.exp().clamp(max=50) + self.activation_bias).sigmoid()
            if self.cfg.variant == 'conjunction':
                includes = membership.amax(1)
                groups = 1 - includes[None] * (1 - activations[:, None])
                mask = (includes > 0).to(groups.dtype)[None]
                scores = self.logic.conjunction(groups, mask)
                group_scores = groups.masked_fill(mask == 0, 1)
            else:
                scores, group_scores = self.logic(activations, membership)
                if fixed_forms is not None:
                    mask = (membership.sum(-1) > 0).to(group_scores.dtype)
                    group_scores = torch.where(mask[None] > 0, group_scores, torch.ones_like(group_scores))
                    scores = self.logic.conjunction(group_scores, mask[None].expand_as(group_scores))
        return {'scores': scores, 'groups': group_scores, 'activations': activations,
                'membership': membership, 'probabilities': probabilities}

    def grounding_loss(self, element_vectors: torch.Tensor) -> torch.Tensor:
        bank = self.private_bank.flatten(0, 1) if self.cfg.variant == 'per_section' else self.bank
        similarities = F.normalize(element_vectors, dim=-1) @ F.normalize(bank, dim=-1).T
        return (1 - similarities.amax(-1)).mean()


def form_alignment(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    distances = (left[:, :, None, :] - right[:, None, :, :]).square().mean(-1)
    return (distances.amin(1).mean() + distances.amin(2).mean()) / 2


def classification_loss(scores: torch.Tensor, targets: torch.Tensor, kind: str, positive_weight=None) -> torch.Tensor:
    scores = scores.float().clamp(1e-6, 1 - 1e-6)
    positive = -targets * scores.log()
    negative = -(1 - targets) * torch.log1p(-scores)
    if kind == 'pos_weight':
        positive = positive * positive_weight
    elif kind == 'focal':
        positive = positive * (1 - scores).square()
        negative = negative * scores.square()
    elif kind == 'asymmetric':
        negative_scores = (scores - 0.05).clamp_min(0)
        negative = -(1 - targets) * negative_scores.pow(4) * torch.log1p(-negative_scores)
    return (positive + negative).mean()
