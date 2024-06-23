from torch import nn, Tensor
from jaxtyping import Float


class ContrastProbe(nn.Module):

    def __init__(
        self,
        d_model: int=4096,
        d_out: int=1,
        device: str="cuda"
    ):
        super(ContrastProbe, self).__init__()
        self.d_model = d_model
        self.d_out = d_out
        self.device = device

        self.probe = nn.Linear(in_features=self.d_model, out_features=self.d_out, bias=True, device=self.device)

    def forward(
        self,
        x: Float[Tensor, "batch d_model"]
    ) -> Float[Tensor, "d_out"]:
        return self.probe(x)