import torch
from torch import nn

from opinion_guided_model_v1 import OpinionGuidedModel


class OpinionAwareModel(OpinionGuidedModel):
    def __init__(
        self, encoder, means=None,
        start_id=None, end_id=None, sep_id=102
    ):
        super().__init__(encoder, means)

        if start_id is None or end_id is None:
            raise ValueError("Aspect marker token IDs are required")

        self.start_id = int(start_id)
        self.end_id = int(end_id)
        self.sep_id = int(sep_id)

        hidden_size = encoder.config.hidden_size

        self.opinion_fusion = nn.Linear(
            hidden_size, hidden_size, bias=False
        )

        # Initially identical to CLS-only regression
        nn.init.zeros_(self.opinion_fusion.weight)

    def forward(
        self, return_aux=False,
        return_opinion=False, **tokens
    ):
        hidden = self.encoder(**tokens).last_hidden_state
        cls = hidden[:, 0, :]

        opinion_logits = self.opinion_head(
            hidden
        ).squeeze(-1)

        ids = tokens["input_ids"]

        # Only sentence-segment tokens can be opinions.
        valid = (
            tokens["attention_mask"].bool()
            & tokens["token_type_ids"].eq(1)
            & ids.ne(self.start_id)
            & ids.ne(self.end_id)
            & ids.ne(self.sep_id)
        )

        # Differentiable predicted-opinion pooling.
        probabilities = torch.sigmoid(
            opinion_logits.float()
        )

        weights = probabilities.square() * valid.float()
        denominator = weights.sum(dim=1, keepdim=True)

        context = (
            hidden.float() * weights.unsqueeze(-1)
        ).sum(dim=1) / denominator.clamp_min(1e-8)

        context = torch.where(
            denominator.gt(1e-8),
            context,
            cls.float()
        )

        delta = (context - cls.float()).to(cls.dtype)

        fused = cls + self.opinion_fusion(delta)

        values = torch.cat(
            (self.valence(fused), self.arousal(fused)),
            dim=-1
        )

        if return_opinion:
            return (
                values,
                self.arousal_class(fused),
                opinion_logits
            )

        if return_aux:
            return values, self.arousal_class(fused)

        return values
