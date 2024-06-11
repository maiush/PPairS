from torch import nn, Tensor
from jaxtyping import Float


class CompareClassifier(nn.Module):

    def __init__(
        self,
        d_model: int=4096,
        device: str="cuda"
    ):
        super(CompareClassifier, self).__init__()
        self.d_model = d_model
        self.device = device

        self.map = nn.Linear(in_features=self.d_model, out_features=1, bias=True, device=self.device)

    def forward(
        self,
        x: Float[Tensor, "batch d_model"]
    ) -> Float[Tensor, "1"]:
        logits = self.map(x)
        return logits