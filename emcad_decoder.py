# Jittor version of EMCAD decoder and its modules
import jittor as jt
import jittor.nn as nn

# ========== UTILS ===========
def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    act = act.lower()
    if act == 'relu':
        return nn.ReLU()
    elif act == 'leakyrelu':
        return nn.LeakyReLU(neg_slope)
    elif act == 'prelu':
        return nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'hswish':
        return nn.Hardswish()
    else:
        raise NotImplementedError(f'activation layer [{act}] not found')

def channel_shuffle(x, groups):
    B, C, H, W = x.shape
    channels_per_group = C // groups
    x = x.view(B, groups, channels_per_group, H, W)
    x = x.transpose(1, 2).contiguous()
    x = x.view(B, C, H, W)
    return x


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# ========== MODULES ===========
class MSDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu', dw_parallel=True):
        super().__init__()
        self.dw_parallel = dw_parallel
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv(in_channels, in_channels, k, stride, padding=k//2, groups=in_channels, bias=False),
                nn.BatchNorm(in_channels),
                act_layer(activation)
            ) for k in kernel_sizes
        ])

    def execute(self, x):
        outs = []
        for dwconv in self.dwconvs:
            out = dwconv(x)
            outs.append(out)
            if not self.dw_parallel:
                x = x + out
        return outs

class MSCB(nn.Module):
    def __init__(self, in_c, out_c, stride, kernel_sizes=[1,3,5], expansion_factor=2, dw_parallel=True, add=True, activation='relu'):
        super().__init__()
        self.add = add
        self.exp_c = in_c * expansion_factor
        self.pw1 = nn.Sequential(
            nn.Conv(in_c, self.exp_c, 1),
            nn.BatchNorm(self.exp_c),
            act_layer(activation)
        )
        self.msdc = MSDC(self.exp_c, kernel_sizes, stride, activation, dw_parallel)
        self.pw2 = nn.Sequential(
            nn.Conv(self.exp_c if add else self.exp_c * len(kernel_sizes), out_c, 1),
            nn.BatchNorm(out_c)
        )
        self.skip_proj = None
        if stride == 1 and in_c != out_c:
            self.skip_proj = nn.Conv(in_c, out_c, 1)

    def execute(self, x):
        out = self.pw1(x)
        outs = self.msdc(out)
        if self.add:
            out = sum(outs)
        else:
            out = jt.concat(outs, dim=1)
        out = channel_shuffle(out, gcd(out.shape[1], self.pw2[0].out_channels))
        out = self.pw2(out)
        if self.skip_proj is not None:
            x = self.skip_proj(x)
        if x.shape == out.shape:
            return x + out
        return out

def MSCBLayer(in_c, out_c, n=1, **kwargs):
    layers = [MSCB(in_c, out_c, **kwargs)]
    for _ in range(1, n):
        layers.append(MSCB(out_c, out_c, stride=1, **kwargs))
    return nn.Sequential(*layers)

class EUCB(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, activation='relu'):
        super().__init__()
        self.dwc = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv(in_c, in_c, k, s, padding=k//2, groups=in_c, bias=False),
            nn.BatchNorm(in_c),
            act_layer(activation)
        )
        self.pwc = nn.Conv(in_c, out_c, 1)

    def execute(self, x):
        x = self.dwc(x)
        x = channel_shuffle(x, x.shape[1])
        return self.pwc(x)

class LGAG(nn.Module):
    def __init__(self, F_g, F_l, F_int, ks=3, groups=1, activation='relu'):
        super().__init__()
        self.Wg = nn.Sequential(
            nn.Conv(F_g, F_int, ks, padding=ks//2, groups=groups),
            nn.BatchNorm(F_int)
        )
        self.Wx = nn.Sequential(
            nn.Conv(F_l, F_int, ks, padding=ks//2, groups=groups),
            nn.BatchNorm(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv(F_int, 1, 1),
            nn.BatchNorm(1),
            nn.Sigmoid()
        )
        self.act = act_layer(activation)

    def execute(self, g, x):
        g1 = self.Wg(g)
        x1 = self.Wx(x)
        psi = self.act(g1 + x1)
        return x * self.psi(psi)




class MaxPool2d(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def execute(self, x):
        return nn.pool(x, kernel_size=self.kernel_size, stride=self.stride, op='maximum')


def pool2d():
    return MaxPool2d(kernel_size=2, stride=2)





class CAB(nn.Module):
    def __init__(self, in_c, ratio=16, activation='relu'):
        super().__init__()
        r_c = max(1, in_c // ratio)
        self.shared = nn.Sequential(
            nn.Conv(in_c, r_c, 1),
            act_layer(activation),
            nn.Conv(r_c, in_c, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def execute(self, x):
        avg = nn.AdaptiveAvgPool2d(1)(x)
        maxp = nn.AdaptiveMaxPool2d(1)(x)
        return self.sigmoid(self.shared(avg) + self.shared(maxp))

class SAB(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv(2, 1, k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def execute(self, x):
        avg = jt.mean(x, dim=1, keepdims=True)
        # maxp = jt.max(x, dim=1, keepdims=True)[0]
        maxp = jt.max(x, dim=1, keepdims=True)
        out = jt.concat([avg, maxp], dim=1)
        return self.sigmoid(self.conv(out))

class EMCAD(nn.Module):
    def __init__(self, channels, kernel_sizes=[1,3,5], expansion_factor=6, dw_parallel=True, add=True, lgag_ks=3, activation='relu'):
        super().__init__()
        self.mscb4 = MSCBLayer(channels[0], channels[0], n=1, stride=1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)
        self.eucb3 = EUCB(channels[0], channels[1])
        self.lgag3 = LGAG(channels[1], channels[1], channels[1]//2, lgag_ks, channels[1]//2)
        self.mscb3 = MSCBLayer(channels[1], channels[1], n=1, stride=1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)
        self.eucb2 = EUCB(channels[1], channels[2])
        self.lgag2 = LGAG(channels[2], channels[2], channels[2]//2, lgag_ks, channels[2]//2)
        self.mscb2 = MSCBLayer(channels[2], channels[2], n=1, stride=1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)
        self.eucb1 = EUCB(channels[2], channels[3])
        self.lgag1 = LGAG(channels[3], channels[3], channels[3]//2, lgag_ks, channels[3]//2)
        self.mscb1 = MSCBLayer(channels[3], channels[3], n=1, stride=1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)
        self.cab4 = CAB(channels[0])
        self.cab3 = CAB(channels[1])
        self.cab2 = CAB(channels[2])
        self.cab1 = CAB(channels[3])
        self.sab = SAB()

    def execute(self, x, skips):
        d4 = self.sab(self.cab4(x)*x)*self.cab4(x)*x
        d4 = self.mscb4(d4)
        d3 = self.eucb3(d4)
        d3 = d3 + self.lgag3(d3, skips[0])
        d3 = self.sab(self.cab3(d3)*d3)*self.cab3(d3)*d3
        d3 = self.mscb3(d3)
        d2 = self.eucb2(d3)
        d2 = d2 + self.lgag2(d2, skips[1])
        d2 = self.sab(self.cab2(d2)*d2)*self.cab2(d2)*d2
        d2 = self.mscb2(d2)
        d1 = self.eucb1(d2)
        d1 = d1 + self.lgag1(d1, skips[2])
        d1 = self.sab(self.cab1(d1)*d1)*self.cab1(d1)*d1
        d1 = self.mscb1(d1)
        return [d4, d3, d2, d1]


__all__=["EMCAD"]