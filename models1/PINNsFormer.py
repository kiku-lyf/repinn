# # implementation of PINNsformer
# # paper: PINNsFormer: A Transformer-Based Framework For Physics-Informed Neural Networks
# # link: https://arxiv.org/abs/2307.11833
#
import torch
import torch.nn as nn
import pdb
from util import get_clones

#
# class WaveAct(nn.Module):
#     def __init__(self):
#         super(WaveAct, self).__init__()
#         self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
#         self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)
#
#     def forward(self, x):
#         return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)
#
#
# class FeedForward(nn.Module):
#     def __init__(self, d_model, d_ff=256):
#         super(FeedForward, self).__init__()
#         self.linear = nn.Sequential(*[
#             nn.Linear(d_model, d_ff),
#             WaveAct(),
#             nn.Linear(d_ff, d_ff),
#             WaveAct(),
#             nn.Linear(d_ff, d_model)
#         ])
#
#     def forward(self, x):
#         return self.linear(x)
#
#
# class EncoderLayer(nn.Module):
#     def __init__(self, d_model, heads):
#         super(EncoderLayer, self).__init__()
#
#         self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
#         self.ff = FeedForward(d_model)
#         self.act1 = WaveAct()
#         self.act2 = WaveAct()
#
#     def forward(self, x):
#         x2 = self.act1(x)
#         x = x + self.attn(x2, x2, x2)[0]
#         x2 = self.act2(x)
#         x = x + self.ff(x2)
#         return x
#
#
# class DecoderLayer(nn.Module):
#     def __init__(self, d_model, heads):
#         super(DecoderLayer, self).__init__()
#
#         self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
#         self.ff = FeedForward(d_model)
#         self.act1 = WaveAct()
#         self.act2 = WaveAct()
#
#     def forward(self, x, e_outputs):
#         x2 = self.act1(x)
#         x = x + self.attn(x2, e_outputs, e_outputs)[0]
#         x2 = self.act2(x)
#         x = x + self.ff(x2)
#         return x
#
#
# class Encoder(nn.Module):
#     def __init__(self, d_model, N, heads):
#         super(Encoder, self).__init__()
#         self.N = N
#         self.layers = get_clones(EncoderLayer(d_model, heads), N)
#         self.act = WaveAct()
#
#     def forward(self, x):
#         for i in range(self.N):
#             x = self.layers[i](x)
#         return self.act(x)
#
#
# class Decoder(nn.Module):
#     def __init__(self, d_model, N, heads):
#         super(Decoder, self).__init__()
#         self.N = N
#         self.layers = get_clones(DecoderLayer(d_model, heads), N)
#         self.act = WaveAct()
#
#     def forward(self, x, e_outputs):
#         for i in range(self.N):
#             x = self.layers[i](x, e_outputs)
#         return self.act(x)
#
#
# class Model(nn.Module):
#     def __init__(self, in_dim, out_dim, hidden_dim, num_layer, hidden_d_ff=512, heads=2):
#         super(Model, self).__init__()
#
#         self.linear_emb = nn.Linear(in_dim, hidden_dim)
#
#         self.encoder = Encoder(hidden_dim, num_layer, heads)
#         self.decoder = Decoder(hidden_dim, num_layer, heads)
#         self.linear_out = nn.Sequential(*[
#             nn.Linear(hidden_dim, hidden_d_ff),
#             WaveAct(),
#             nn.Linear(hidden_d_ff, hidden_d_ff),
#             WaveAct(),
#             nn.Linear(hidden_d_ff, out_dim)
#         ])
#
#     def forward(self, coords):
#         '''
#         PINNsFormer forward
#
#         Args:
#         -----
#             coords : 3D torch.float
#                 inputs, shape (batch, sequence_length, input_dimension)
#                 Expected: (batch, seq_len, 3) where 3 = [x, y, t]
#
#         Returns:
#         --------
#             output : 3D torch.float
#                 outputs, shape (batch, sequence_length, output_dimension)
#                 Returns: (batch, seq_len, 2) where 2 = [psi, p]
#
#         Example
#         -------
#         >>> model = Model(in_dim=3, out_dim=2, hidden_dim=64, num_layer=4, heads=2)
#         >>> x = torch.normal(0, 1, size=(32, 100, 3))  # (batch, seq_len, features)
#         >>> model(x).shape
#         torch.Size([32, 100, 2])
#         '''
#         src = self.linear_emb(coords)
#
#         e_outputs = self.encoder(src)
#         d_output = self.decoder(src, e_outputs)
#         output = self.linear_out(d_output)
#         return output


class WaveAct(nn.Module):
    def __init__(self):
        super(WaveAct, self).__init__()
        self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256):
        super(FeedForward, self).__init__()
        self.linear = nn.Sequential(*[
            nn.Linear(d_model, d_ff),
            WaveAct(),
            nn.Linear(d_ff, d_ff),
            WaveAct(),
            nn.Linear(d_ff, d_model)
        ])

    def forward(self, x):
        return self.linear(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super(EncoderLayer, self).__init__()

        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.ff = FeedForward(d_model)
        self.act1 = WaveAct()
        self.act2 = WaveAct()

    def forward(self, x):
        x2 = self.act1(x)
        x = x + self.attn(x2, x2, x2)[0]
        x2 = self.act2(x)
        x = x + self.ff(x2)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super(DecoderLayer, self).__init__()

        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.ff = FeedForward(d_model)
        self.act1 = WaveAct()
        self.act2 = WaveAct()

    def forward(self, x, e_outputs):
        x2 = self.act1(x)
        x = x + self.attn(x2, e_outputs, e_outputs)[0]
        x2 = self.act2(x)
        x = x + self.ff(x2)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super(Encoder, self).__init__()
        self.N = N
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.act = WaveAct()

    def forward(self, x):
        for i in range(self.N):
            x = self.layers[i](x)
        return self.act(x)


class Decoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super(Decoder, self).__init__()
        self.N = N
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.act = WaveAct()

    def forward(self, x, e_outputs):
        for i in range(self.N):
            x = self.layers[i](x, e_outputs)
        return self.act(x)


class Model(nn.Module):
    def __init__(self, d_out, d_model, d_hidden, N, heads):
        super(Model, self).__init__()

        self.linear_emb = nn.Linear(3, d_model)

        self.encoder = Encoder(d_model, N, heads)
        self.decoder = Decoder(d_model, N, heads)
        self.linear_out = nn.Sequential(*[
            nn.Linear(d_model, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, d_out)
        ])

    def forward(self, x, y, t):
        src = torch.cat((x, y, t), dim=-1)
        src = self.linear_emb(src)
        e_outputs = self.encoder(src)
        d_output = self.decoder(src, e_outputs)
        output = self.linear_out(d_output)
        return output


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
