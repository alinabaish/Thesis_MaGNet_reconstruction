from typing import Optional

from torch import Tensor
from torch_scatter import scatter


# global average pooling operator
# output batch-wise graph-level-outputs by averaging node features across the node dimension
def global_mean_pool(x: Tensor, batch: Optional[Tensor],
                     size: Optional[int] = None) -> Tensor:

    if batch is None:
        return x.mean(dim=-2, keepdim=x.dim() == 2)
    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim=-2, dim_size=size, reduce='mean')


# global maximum pooling operator
# output batch-wise graph-level-outputs by taking the channel-wise maximum across the node dimension
def global_max_pool(x: Tensor, batch: Optional[Tensor],
                    size: Optional[int] = None) -> Tensor:
 
    if batch is None:
        return x.max(dim=-2, keepdim=x.dim() == 2)[0]
    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim=-2, dim_size=size, reduce='max')


