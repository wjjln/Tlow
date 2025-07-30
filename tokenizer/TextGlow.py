import torch
from torch import nn
from torch.nn import functional as F
from math import log, pi
import numpy as np
from scipy import linalg as la

logabs = lambda x: torch.log(torch.abs(x))

def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)

def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


class ActNorm1d(nn.Module):
    def __init__(self, in_dim, logdet=True):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, in_dim))
        self.scale = nn.Parameter(torch.ones(1, in_dim))
        # self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.initialized = False
        self.logdet = logdet

    def initialize(self, input):
        with torch.no_grad():
            mean = torch.mean(input, 0, keepdim=True)
            # std = torch.std(input, 0)
            self.loc.data.copy_(-mean)
            # self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        # if self.initialized.item() == 0:
        if not self.initialized:
            self.initialize(input)
            # self.initialized.fill_(1)
            self.initialized = True

        log_abs = logabs(self.scale)
        logdet = torch.sum(log_abs)

        if self.logdet:
            return self.scale * (input + self.loc), logdet
        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class InvertibleLinearLU(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        weight = np.random.randn(in_dim, in_dim)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        self.register_buffer("w_p", torch.from_numpy(w_p))
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(torch.from_numpy(w_s)))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(torch.from_numpy(w_l))
        self.w_s = nn.Parameter(logabs(torch.from_numpy(w_s)))
        self.w_u = nn.Parameter(torch.from_numpy(w_u))

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )
        return weight

    def forward(self, input):
        weight = self.calc_weight()
        out = F.linear(input, weight)
        logdet = torch.sum(self.w_s)
        return out, logdet

    def reverse(self, output):
        weight = self.calc_weight()
        return F.linear(output, weight.inverse())


class ZeroLinear(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.linear = nn.Linear(in_channel, out_channel)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel))

    def forward(self, input):
        out = self.linear(input)
        out = out * torch.exp(self.scale * 3)
        return out


class AffineCoupling1d(nn.Module):
    def __init__(self, in_dim, filter_size=512, affine=True):
        super().__init__()
        self.affine = affine
        self.in_dim = in_dim
        assert in_dim % 2 == 0, "Input dimension must be even for coupling layer."
        
        self.net = nn.Sequential(
            nn.Linear(in_dim // 2, filter_size),
            # nn.ReLU(inplace=True),
            nn.ReLU(),
            nn.Linear(filter_size, filter_size),
            # nn.ReLU(inplace=True),
            nn.ReLU(),
            ZeroLinear(filter_size, in_dim if self.affine else in_dim // 2),
        )

    def forward(self, input):
        in_a, in_b = input.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(in_a).chunk(2, 1)
            s = torch.exp(log_s)
            out_b = s * (in_b + t)
            logdet = torch.sum(torch.log(s), dim=1)
        else:
            net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = None

        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)
        if self.affine:
            log_s, t = self.net(out_a).chunk(2, 1)
            s = torch.sigmoid(log_s + 2)
            in_b = out_b / s - t
        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out
        return torch.cat([out_a, in_b], 1)


class Flow(nn.Module):
    def __init__(self, in_dim, affine=True):
        super().__init__()
        self.actnorm = ActNorm1d(in_dim)
        self.invlinear = InvertibleLinearLU(in_dim)
        self.coupling = AffineCoupling1d(in_dim, affine=affine)

    def forward(self, input):
        out, logdet_act = self.actnorm(input)
        out, logdet_inv = self.invlinear(out)
        out, logdet_coup = self.coupling(out)
        total_logdet = logdet_act + logdet_inv
        if logdet_coup is not None:
            total_logdet = total_logdet + logdet_coup
        return out, total_logdet

    def reverse(self, output):
        input = self.coupling.reverse(output)
        input = self.invlinear.reverse(input)
        input = self.actnorm.reverse(input)
        return input


class TextFlowBlock(nn.Module):
    def __init__(self, in_dim, n_flow, split=True, affine=True):
        super().__init__()
        
        assert in_dim % 2 == 0, "Input dimension for a block must be even."
        
        self.flows = nn.ModuleList([Flow(in_dim, affine=affine) for _ in range(n_flow)])
        self.split = split

        if split:
            self.prior = ZeroLinear(in_dim // 2, in_dim) 
        else:
            self.prior = ZeroLinear(in_dim, in_dim * 2)

    def forward(self, input):
        b_size = input.shape[0]
        logdet = 0
        out = input

        for flow in self.flows:
            out, det = flow(out)
            logdet = logdet + det

        if self.split:
            out, z_new = out.chunk(2, 1)
            mean, log_sd = self.prior(out).chunk(2, 1)
            log_p = gaussian_log_p(z_new, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
        else:
            z_new = out
            zero = torch.zeros_like(out)
            mean, log_sd = self.prior(zero).chunk(2, 1)
            log_p = gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
            
        return out, logdet, log_p, z_new

    def reverse(self, output, eps, reconstruct=False):
        if reconstruct:
            if self.split:
                input = torch.cat([output, eps], 1)
            else:
                input = eps
        else:
            if self.split:
                mean, log_sd = self.prior(output).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = torch.cat([output, z], 1)
            else:
                zero = torch.zeros_like(output)
                mean, log_sd = self.prior(zero).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = z

        for flow in reversed(self.flows):
            input = flow.reverse(input)

        return input


class TextGlow(nn.Module):
    def __init__(self, in_dim, n_flow, n_block, affine=True):
        super().__init__()

        self.blocks = nn.ModuleList()
        current_dim = in_dim
        for i in range(n_block - 1):
            assert current_dim % 2 == 0, f"Dimension becomes odd at block {i}. Please choose a different in_dim or n_block."
            self.blocks.append(TextFlowBlock(current_dim, n_flow, split=True, affine=affine))
            current_dim //= 2
        
        self.blocks.append(TextFlowBlock(current_dim, n_flow, split=False, affine=affine))

    def forward(self, input):
        log_p_sum = 0
        logdet = 0
        out = input
        z_outs = []

        for block in self.blocks:
            out, det, log_p, z_new = block(out)
            z_outs.append(z_new)
            logdet = logdet + det
            if log_p is not None:
                log_p_sum = log_p_sum + log_p

        return log_p_sum, logdet, z_outs

    def reverse(self, z_list, reconstruct=False):
        input = self.blocks[-1].reverse(z_list[-1], z_list[-1], reconstruct=reconstruct)

        for i, block in enumerate(reversed(self.blocks[:-1])):
            z_for_this_block = z_list[-(i + 2)]
            input = block.reverse(input, z_for_this_block, reconstruct=reconstruct)
            
        return input