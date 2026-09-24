import math
import torch
import torch.nn as nn

class Fusion_Block(nn.Module):
    def __init__(self, args):
        super(Fusion_Block, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.activation = nn.GELU()
        self.conv1 = nn.Conv1d(in_channels=args.d_feature, out_channels=args.d_model, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=args.d_model, out_channels=args.d_feature, kernel_size=1)
        self.norm1 = nn.BatchNorm1d(args.d_feature)
        self.norm2 = nn.BatchNorm1d(args.d_feature)
        self.norm3 = torch.nn.BatchNorm1d(args.d_feature)
        self.fc_out = nn.Linear(args.d_model, args.d_feature, bias=True)
        patch_num = int((args.pred_len - args.patch_len) / args.stride + 1) + 1
        self.linear_out = nn.Linear(patch_num * args.patch_len, args.pred_len)

    def forward(self, x, x_date, y_date):
        B, L, D, k = x_date.shape
        scale = 1. / math.sqrt(k)
        scores = torch.einsum("bldk,bodk->bdlo", x_date, y_date)
        A = torch.softmax(scale * scores, dim=-2)
        V = torch.einsum("bdnl,bdnp->bdpl", x, A)
        B, D, _, _ = V.shape
        x = V.reshape(B, D, -1)
        x = self.linear_out(x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))
        y = self.norm2((x + y)).transpose(-1, -2)
        return y
