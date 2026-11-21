import math
import torch
from torch import nn
from torch.nn import functional as F
from round2_experiment import ArousalModel


class AspectLocalModel(ArousalModel):
    """CLS + aspect-conditioned local context attention."""

    def __init__(self, encoder, means=None,
                 start_id=None, end_id=None, sep_id=102):
        super().__init__(encoder, means)
        self.start_id = int(start_id)
        self.end_id = int(end_id)
        self.sep_id = int(sep_id)

        hidden_size = encoder.config.hidden_size
        self.local_proj = nn.Linear(
            hidden_size, hidden_size, bias=False
        )
        nn.init.zeros_(self.local_proj.weight)

    def forward(self, return_aux=False, **tokens):
        hidden = self.encoder(**tokens).last_hidden_state
        cls = hidden[:, 0, :]
        ids = tokens["input_ids"]

        sentence = tokens["token_type_ids"].eq(1)
        start = ids.eq(self.start_id) & sentence
        end = ids.eq(self.end_id) & sentence

        span_mask = (
            start.long().cumsum(1)
            - end.long().cumsum(1)
        ).gt(0) & ~start & ~end

        count = span_mask.sum(1, keepdim=True)

        aspect = (
            hidden * span_mask.unsqueeze(-1)
        ).sum(1) / count.clamp_min(1)

        valid = (
            start.sum(1).eq(end.sum(1))
            & start.any(1)
            & count.squeeze(1).gt(0)
        )

        aspect = torch.where(
            valid.unsqueeze(-1), aspect, cls
        )

        # 目标方面前后各 8 个 token
        local_mask = F.max_pool1d(
            span_mask.float().unsqueeze(1),
            kernel_size=17, stride=1, padding=8
        ).squeeze(1).bool()

        local_mask &= sentence
        local_mask &= tokens["attention_mask"].bool()
        local_mask &= ~start & ~end
        local_mask &= ids.ne(self.sep_id)

        # 没有完整方面标记时，安全回退到 CLS
        valid = valid & local_mask.any(1)
        fallback = torch.zeros_like(local_mask)
        fallback[:, 0] = True
        local_mask = torch.where(
            valid.unsqueeze(1), local_mask, fallback
        )

        scores = (
            (hidden * aspect.unsqueeze(1)).sum(-1).float()
            / math.sqrt(hidden.shape[-1])
        )
        scores = scores.masked_fill(~local_mask, -1e4)

        weights = torch.softmax(scores, dim=1)
        context = (
            hidden * weights.to(hidden.dtype).unsqueeze(-1)
        ).sum(1)

        fused = cls + self.local_proj(context - cls)

        values = torch.cat((
            self.valence(fused),
            self.arousal(fused)
        ), dim=-1)

        if return_aux:
            return values, self.arousal_class(fused)
        return values
