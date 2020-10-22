
import torch
import torch.nn as nn

def seq_c_b_a_s(x, conv, relu, bn, skip=None, skip_enable=False):
    out = conv(x)
    out = bn(out)
    out = relu(out)
    if skip_enable:
        out += skip
    return out

def seq_c_b_s_a(x, conv, relu, bn, skip=None, skip_enable=False):
    out = conv(x)
    out = bn(out)
    if skip_enable:
        out += skip
    out = relu(out)
    return out

def seq_c_a_b_s(x, conv, relu, bn, skip=None, skip_enable=False):
    out = conv(x)
    out = relu(out)
    out = bn(out)
    if skip_enable:
        out += skip
    return out

def seq_b_c_a_s(x, conv, relu, bn, skip=None, skip_enable=False):
    out = bn(x)
    out = conv(out)
    out = relu(out)
    if skip_enable:
        out += skip
    return out

def seq_b_a_c_s(x, conv, relu, bn, skip=None, skip_enable=False):
    out = bn(x)
    out = relu(out)
    out = conv(out)
    if skip_enable:
        out += skip
    return out

class SliceBN(nn.Module):
    def __init__(self, channel, group):
        super(SliceBN, self).__init__()
        self.group = group
        self.bn = nn.ModuleList([nn.BatchNorm2d(channel) for i in range(group * group)])

    def forward(self, x):
        group = self.group
        if group == 1:
            return self.bn[0](x)
        else:
            shape = x.shape
            x = x.reshape(shape[0], shape[1], group, shape[2] // group, group, shape[3] // group)
            y = x.new_zeros(x.shape)
            for i in range(self.group):
                for j in range(self.group):
                    slices = x[:,:, i, :, j, :]
                    slices = self.bn[i*group+j](slices)
                    y[:,:, i, :, j, :] = slices
            y = y.reshape(shape)
            return y

class FrozenBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)

    def forward(self, x):
        scale = self.weight * (self.running_var + self.eps).rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias

class StaticBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(num_features), requires_grad=True)
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)

    def forward(self, x):
        scale = self.weight * (self.running_var + self.eps).rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias

def norm(channel, args=None, feature_stride=None):
    keyword = None
    if args is not None:
        keyword = getattr(args, "keyword", None)

    if keyword is None:
        return nn.BatchNorm2d(channel)

    if "group-norm" in keyword:
        group = getattr(args, "fm_quant_group", 2)
        return nn.GroupNorm(group, channel)

    if "static-bn" in keyword:
        return StaticBatchNorm2d(channel)

    if "freeze-bn" in keyword:
        return FrozenBatchNorm2d(channel)

    return nn.BatchNorm2d(channel)

def actv(args=None):
    keyword = None
    if args is not None:
        keyword = getattr(args, "keyword", None)

    if keyword is None:
        return nn.ReLU(inplace=True)

    if 'PReLU' in keyword:
        return nn.PReLU()

    if 'NReLU' in keyword:
        return nn.Sequential()

    return nn.ReLU(inplace=True)

