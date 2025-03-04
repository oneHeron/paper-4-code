import torch
import torch.nn as nn
import torch.nn.functional as F

from sparse_layer import SpGATLayer


class SpGAT(nn.Module):
    def __init__(self, num_features, B_dim, hidden_size, embedding_size, alpha):
        super(SpGAT, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.alpha = alpha
        # Encoder
        self.conv1 = SpGATLayer(num_features, hidden_size, alpha)
        # self.transformer1 = TransformerLayer(hidden_size, num_heads, num_layers, dropout)
        self.conv2 = SpGATLayer(hidden_size, embedding_size, alpha)

        self.conv3 = SpGATLayer(B_dim, hidden_size, alpha)
        # self.transformer2 = TransformerLayer(hidden_size, num_heads, num_layers, dropout)
        self.conv4 = SpGATLayer(hidden_size, embedding_size, alpha)
        # Decoder
        self.conv5 = SpGATLayer(embedding_size, hidden_size, alpha)
        # self.transformer3 = TransformerLayer(hidden_size, num_heads, num_layers, dropout)
        self.conv6 = SpGATLayer(hidden_size, num_features, alpha)

        self.conv7 = SpGATLayer(embedding_size, hidden_size, alpha)
        # self.transformer4 = TransformerLayer(hidden_size, num_heads, num_layers, dropout)
        self.conv8 = SpGATLayer(hidden_size, B_dim, alpha)

    # def forward(self, x, B, adj, M):
    #     h = self.conv1(x, adj, M)
    #     # h = self.transformer1(h)
    #     z1 = self.conv2(h, adj, M)
    #
    #     b = self.conv3(B, adj, M)
    #     # b = self.transformer2(b)
    #     z2 = self.conv4(b, adj, M)
    #
    #     z = (z1 + z2) / 2
    #
    #     z = F.normalize(z, p=2, dim=1)
    #
    #     x_hat, B_hat, x_hat2, B_hat2 = self.decode(z1, z2, z, adj, M)
    #
    #     A_pred = self.dot_product_decode(z)
    #     return A_pred, z, x_hat, B_hat, x_hat2, B_hat2

    def forward(self, x, B, adj):
        h = self.conv1(x, adj)
        # h = self.transformer1(h)
        z1 = self.conv2(h, adj)

        b = self.conv3(B, adj)
        # b = self.transformer2(b)
        z2 = self.conv4(b, adj)

        z = (z1 + z2) / 2

        z = F.normalize(z, p=2, dim=1)

        x_hat, B_hat, x_hat2, B_hat2 = self.decode(z1, z2, z, adj)

        A_pred = self.dot_product_decode(z)
        return A_pred, z, x_hat, B_hat, x_hat2, B_hat2

    # Structure Decoder
    def dot_product_decode(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred

    # Attribute Decoder
    # def decode(self, z1, z2, z, adj, M):
    #     z1 = self.conv5(z1, adj, M)
    #     # z1 = self.transformer3(z1)
    #     x_hat = self.conv6(z1, adj, M)
    #
    #     h = self.conv5(z, adj, M)
    #     # h = self.transformer3(h)
    #     x_hat2 = self.conv6(h, adj, M)
    #
    #     z2 = self.conv7(z2, adj, M)
    #     # z2 = self.transformer4(z2)
    #     B_hat = self.conv8(z2, adj, M)
    #
    #     h = self.conv7(z, adj, M)
    #     # h = self.transformer4(h)
    #     B_hat2 = self.conv8(h, adj, M)
    #     return x_hat, B_hat, x_hat2, B_hat2

    def decode(self, z1, z2, z, adj):
        z1 = self.conv5(z1, adj)
        # z1 = self.transformer3(z1)
        x_hat = self.conv6(z1, adj)

        h = self.conv5(z, adj)
        # h = self.transformer3(h)
        x_hat2 = self.conv6(h, adj)

        z2 = self.conv7(z2, adj)
        # z2 = self.transformer4(z2)
        B_hat = self.conv8(z2, adj)

        h = self.conv7(z, adj)
        # h = self.transformer4(h)
        B_hat2 = self.conv8(h, adj)
        return x_hat, B_hat, x_hat2, B_hat2


class Clustering_Module(nn.Module):

    def __init__(self, input_dim, num_clusters, use_bias=False):
        super(Clustering_Module, self).__init__()
        self.input = input_dim
        self.num_clusters = num_clusters
        self.Id = torch.eye(num_clusters)

        self.gamma = nn.Sequential(
            nn.Linear(input_dim, num_clusters, bias=use_bias),
            nn.Softmax(dim=1)
        )
        self.mu = nn.Sequential(
            nn.Linear(num_clusters, input_dim, bias=use_bias),
        )

    def _mu(self):
        return self.mu(self.Id)

    def predict(self, x):
        return self.gamma(x).argmax(-1)

    def predict_proba(self, x):
        return self.gamma(x)

    def forward(self, x):
        g = self.gamma(x)
        tx = self.mu(g)
        u = self.mu[0].weight.T
        return x, g, tx, u


class Clustering_Module_Loss(nn.Module):
    def __init__(self, num_clusters, alpha=1, lbd=1, orth=False, normalize=True, device='cuda'):
        super(Clustering_Module_Loss, self).__init__()

        if hasattr(alpha, '__iter__'):
            self.alpha = torch.tensor(alpha, device=device)
            # Use this if alpha are not all the same
            # self.alpha = torch.sort(self.alpha).values
        else:
            self.alpha = torch.ones(num_clusters, device=device) * alpha

        self.lbd = lbd
        self.orth = orth
        self.normalize = normalize
        self.Id = torch.eye(num_clusters, device=device)
        self.mask = 1 - torch.eye(num_clusters, device=device)

    def forward(self, inputs, targets=None, split=False):
        x, g, tx, u = inputs
        n, d = x.shape
        k = g.shape[1]

        nd = (n * d) if self.normalize else 1.

        loss_E1 = torch.sum(torch.square(x - tx)) / nd

        if self.orth:
            loss_E2 = torch.sum(g * (1 - g)) / nd

            uu = torch.matmul(u, u.T)
            loss_E3 = torch.sum(torch.square(uu - self.Id.to(uu.device))) * self.lbd
        else:
            loss_E2 = torch.sum(torch.sum(g * (1 - g), 0) * torch.sum(torch.square(u), 1)) / nd

            gg = torch.matmul(g.T, g)
            uu = torch.matmul(u, u.T)
            gu = gg * uu
            gu = gu * self.mask
            loss_E3 = - torch.sum(gu) / nd

        lmg = torch.log(torch.mean(g, 0) + 1e-10)
        # Use this if alpha are not all the same
        # lmg = torch.sort(lmg).values
        loss_E4 = lmg

        if split:
            nd = 1. if self.normalize else n * d
            return torch.stack(
                (loss_E1 / nd, loss_E2 / nd, loss_E3 / (1 if self.orth else nd), torch.sum(loss_E4 * (1 - self.alpha))))
        else:
            return loss_E1 + loss_E2 + loss_E3 + torch.sum(loss_E4 * (1 - self.alpha))
