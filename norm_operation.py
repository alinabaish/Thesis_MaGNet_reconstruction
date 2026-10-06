import torch
from torch import Tensor
from torch_scatter import scatter
from torch_geometric.typing import OptTensor
from typing import Optional


# Pair normalization over node features as described in the paper <https://arxiv.org/abs/1909.12223>
class PairNorm(torch.nn.Module):

    def __init__(self, scale: float = 1., scale_individually: bool = False,
                 eps: float = 1e-5):
        super().__init__()

        self.scale = scale
        self.scale_individually = scale_individually
        self.eps = eps

    def forward(self, x: Tensor, batch: OptTensor = None) -> Tensor:
        """"""
        scale = self.scale

        # compute pair norm and divide by the norm 
        if batch is None:
            x = x - x.mean(dim=0, keepdim=True)

            if not self.scale_individually:
                return scale * x / (self.eps + x.pow(2).sum(-1).mean()).sqrt()
            else:
                return scale * x / (self.eps + x.norm(2, -1, keepdim=True))

        else:
            mean = scatter(x, batch, dim=0, reduce='mean')
            x = x - mean.index_select(0, batch)

            if not self.scale_individually:
                return scale * x / torch.sqrt(self.eps + scatter(
                    x.pow(2).sum(-1, keepdim=True), batch, dim=0,
                    reduce='mean').index_select(0, batch))
            else:
                return scale * x / (self.eps + x.norm(2, -1, keepdim=True))

    # parameter configure 
    def __repr__(self):
        return f'{self.__class__.__name__}()'
    
# Layer normalization by subtracting the mean from the inputs in the paper <https://arxiv.org/pdf/2003.13663.pdf>
class MeanSubtractionNorm(torch.nn.Module):
    def reset_parameters(self):
        pass

    def forward(self, x: Tensor, batch: Optional[Tensor] = None,
                dim_size: Optional[int] = None) -> Tensor:
        """"""

        # compute the embedding via substracting mean 
        if batch is None:
            return x - x.mean(dim=0, keepdim=True)

        mean = scatter(x, batch, dim=0, dim_size=dim_size, reduce='mean')
        return x - mean[batch]

    # model parameter configuration 
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
