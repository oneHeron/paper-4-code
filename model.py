import torch
import torch.nn as nn
import torch.nn.functional as F

from layer import GATLayer


class GAT(nn.Module):
    def __init__(self, num_features, B_dim, hidden_size, embedding_size, alpha):
        super(GAT, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.alpha = alpha
        # Encoder
        self.conv1 = GATLayer(num_features, hidden_size, alpha)
        self.conv2 = GATLayer(hidden_size, embedding_size, alpha)

        self.conv3 = GATLayer(B_dim, hidden_size, alpha)
        self.conv4 = GATLayer(hidden_size, embedding_size, alpha)
        # Decoder
        self.conv5 = GATLayer(embedding_size, hidden_size, alpha)
        self.conv6 = GATLayer(hidden_size, num_features, alpha)

        self.conv7 = GATLayer(embedding_size, hidden_size, alpha)
        self.conv8 = GATLayer(hidden_size, B_dim, alpha)

    def forward(self, x, B, adj, M):
        h = self.conv1(x, adj, M)
        z1 = self.conv2(h, adj, M)

        b = self.conv3(B, adj, M)
        z2 = self.conv4(b, adj, M)

        z = (z1 + z2) / 2

        z = F.normalize(z, p=2, dim=1)

        x_hat, B_hat, x_hat2, B_hat2 = self.decode(z1, z2, z, adj, M)
        A_pred = self.dot_product_decode(z)
        return A_pred, z, x_hat, B_hat, x_hat2, B_hat2

    # Structure Decoder
    def dot_product_decode(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred

    # Attribute Decoder
    def decode(self, z1, z2, z, adj, M):
        z1 = self.conv5(z1, adj, M)
        x_hat = self.conv6(z1, adj, M)

        h = self.conv5(z, adj, M)
        x_hat2 = self.conv6(h, adj, M)

        z2 = self.conv7(z2, adj, M)
        B_hat = self.conv8(z2, adj, M)

        h = self.conv7(z, adj, M)
        B_hat2 = self.conv8(h, adj, M)
        return x_hat, B_hat, x_hat2, B_hat2
