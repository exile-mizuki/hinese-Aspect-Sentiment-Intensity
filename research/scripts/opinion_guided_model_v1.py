from torch import nn

from round2_experiment import ArousalModel


class OpinionGuidedModel(ArousalModel):
    def __init__(self, encoder, means=None):
        super().__init__(encoder, means)

        hidden_size = encoder.config.hidden_size
        self.opinion_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        return_aux=False,
        return_opinion=False,
        **tokens
    ):
        hidden = self.encoder(**tokens).last_hidden_state
        cls = hidden[:, 0, :]

        values = __import__("torch").cat(
            (
                self.valence(cls),
                self.arousal(cls)
            ),
            dim=-1
        )

        if return_opinion:
            aux = self.arousal_class(cls)
            opinion_logits = self.opinion_head(
                hidden
            ).squeeze(-1)

            return values, aux, opinion_logits

        if return_aux:
            return values, self.arousal_class(cls)

        return values
