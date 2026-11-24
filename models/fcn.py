import numpy as np
import torch.nn as nn
import torch


def select_act_fn(act_fn):
    return {
        "relu": nn.ReLU()
    }[act_fn]


def select_norm_layer(norm_type, dim):
    return {
        "layer_norm": nn.LayerNorm(dim),
        "batch_norm": nn.BatchNorm1d(dim)
    }.get(norm_type, nn.Identity())


class Block(nn.Module):

    def __init__(self, input_dim, output_dim, bias, act_fn, norm_type, residual_stream, block_type):
        super().__init__()
        self.residual_stream = residual_stream
        self.block_type = block_type
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.act_fc = select_act_fn(act_fn)
        self.norm = select_norm_layer(norm_type, input_dim if block_type == "pre" else output_dim)

    def forward(self, x):
        h = x
        if self.block_type == "pre":
            h = self.norm(h)
        h = self.linear(h)
        if self.block_type == "pre_act":
            h = self.norm(h)
        h = self.act_fc(h)
        if self.residual_stream:
            h = h + x
        if self.block_type == "post":
            h = self.norm(h)
        return h


class FCN(nn.Module):

    def __init__(
        self,
        depth,
        input_dim,
        output_dim,
        hidden_dim,
        bias,
        act_fn,
        norm_type,
        residual_stream,
        init_scale,
        block_type,
        checkpoint_path=None
    ):
        super().__init__()
        assert depth > 1

        first_layer = Block(
            input_dim,
            hidden_dim,
            act_fn=act_fn,
            norm_type=norm_type,
            residual_stream=False,
            bias=bias,
            block_type=block_type,
        )

        layers = [first_layer]
        for _ in range(depth - 2):
            layers.append(
                Block(
                    hidden_dim,
                    hidden_dim,
                    act_fn=act_fn,
                    norm_type=norm_type,
                    residual_stream=residual_stream,
                    bias=bias,
                    block_type=block_type,
                )
            )

        layers.append(nn.Linear(hidden_dim, output_dim, bias=bias))
        self.layers = nn.Sequential(*layers)
        self.scale_weights(init_scale)

    def scale_weights(self, init_scale):
        for layer in self.layers:
            if type(layer) == nn.Linear:
                layer.weight.data *= init_scale
            elif type(layer) == Block:
                layer.linear.weight.data *= init_scale
            else:
                raise Exception("Invalid layer")

    def forward(self, x):
        return self.layers(x)