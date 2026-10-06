
from typing import Optional
from torch import Tensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import OptTensor

# construct the linearized graph convolutional layer 

class MaGNetConv(MessagePassing):

    _cached_x: Optional[Tensor]

    def __init__(self, in_channels, out_channels, K: int = 1, cached: bool = False, add_self_loops: bool = True,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        # convolutional hop
        self.cached = cached
        self.add_self_loops = add_self_loops
        self._cached_x = None
        self.K = K
        
        # input and output embedding dims 
        self.in_channels = in_channels
        self.out_channels = out_channels

        # perform linear transformation 
        self.lin = Linear(in_channels, out_channels, bias=bias)
        self.reset_parameters()

    # messaging step 
    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    # aggregation step 
    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)

    # model transformation and parameter updating 
    def forward(self, x, edge_index, edge_weight: OptTensor = None):
        cache = self._cached_x
        
        # determine the unweighted computation graph 
        if cache is None:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = gcn_norm( 
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops, self.flow, dtype=x.dtype)
            elif isinstance(edge_index, SparseTensor):
                edge_index = gcn_norm(  
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops, self.flow, dtype=x.dtype)
                
        # gradient  backprogation 
            for _ in range(self.K):
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                                   size=None)
                if self.cached:
                    self._cached_x = x
        else:
            x = cache.detach()

        return self.lin(x)
    
    # configuration of input and output channels 
    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={self.K})')

    # reset the layer weight parameter 
    def reset_parameters(self):
        self.lin.reset_parameters()
        self._cached_x = None
