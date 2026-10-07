"""Neural network layers for encoders."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / fan_in ** 0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, input_mask):
        """
        x: (batch_size, in_features)
        input_mask: (in_features,) boolean vector
        """
        assert input_mask.dim() == 1 and input_mask.shape[0] == self.in_features
        # Subset inputs and corresponding weights
        x_selected = x[:, input_mask]
        w_selected = self.weight[:, input_mask]  # (out_features, selected_in)
        return F.linear(x_selected, w_selected, self.bias)
