import torch
import torch.nn as nn
import torch.nn.functional as F


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropagation layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGATLayer(nn.Module):
    """
    Sparse GAT layer with an additional structural matrix M.
    """

    def __init__(self, in_features, out_features, alpha, concat=True):
        super(SpGATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a_self = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_self.data, gain=1.414)

        self.a_neighs = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_neighs.data, gain=1.414)

        # self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input_, adj):
        dv = 'cuda' if input_.is_cuda else 'cpu'
        N = input_.size()[0]

        edge = adj.nonzero().t()

        h = torch.mm(input_, self.W)  # 变换特征
        assert not torch.isnan(h).any()

        # 计算注意力得分（受M的影响）
        attn_self = torch.mm(h, self.a_self)  # (N, 1)
        attn_neighs = torch.mm(h, self.a_neighs)  # (N, 1)
        attn_dense = attn_self + attn_neighs.T  # (N, N)
        attn_dense = self.leakyrelu(attn_dense)

        # 只对邻接矩阵中有连接的部分计算注意力
        edge_e = attn_dense[edge[0], edge[1]]
        edge_e = torch.exp(-edge_e)
        assert not torch.isnan(edge_e).any()

        # 计算归一化因子
        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))

        # 应用 dropout
        # edge_e = self.dropout(edge_e)

        # 聚合信息
        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()

        # 归一化
        h_prime = h_prime.div(e_rowsum)
        assert not torch.isnan(h_prime).any()

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + f" ({self.in_features} -> {self.out_features})"
