import torch
from torch import nn
from round2_experiment import ArousalModel

class AspectPoolingModel(ArousalModel):
    def __init__(self, encoder, means=None, start_id=None, end_id=None):
        super().__init__(encoder, means)
        self.start_id = int(start_id)
        self.end_id = int(end_id)
        hidden = encoder.config.hidden_size
        self.span_proj = nn.Linear(hidden, hidden, bias=False)
        nn.init.zeros_(self.span_proj.weight)

    def forward(self, return_aux=False, **tokens):
        hidden = self.encoder(**tokens).last_hidden_state
        cls = hidden[:, 0, :]
        ids = tokens["input_ids"]

        start = ids.eq(self.start_id)
        end = ids.eq(self.end_id)

        if "token_type_ids" in tokens:
            sentence = tokens["token_type_ids"].eq(1)
            start = start & sentence
            end = end & sentence

        inside = (
            start.long().cumsum(dim=1)
            - end.long().cumsum(dim=1)
        ) > 0
        inside = inside & ~start & ~end

        if "attention_mask" in tokens:
            inside = inside & tokens["attention_mask"].bool()

        count = inside.sum(dim=1, keepdim=True)
        span = (
            hidden * inside.unsqueeze(-1).to(hidden.dtype)
        ).sum(dim=1) / count.clamp_min(1)

        valid = (
            start.sum(dim=1).eq(end.sum(dim=1))
            & start.any(dim=1)
            & count.squeeze(1).gt(0)
        )
        span = torch.where(valid.unsqueeze(1), span, cls)

        representation = cls + self.span_proj(span - cls)
        values = torch.cat((
            self.valence(representation),
            self.arousal(representation)
        ), dim=-1)

        if return_aux:
            return values, self.arousal_class(representation)
        return values
