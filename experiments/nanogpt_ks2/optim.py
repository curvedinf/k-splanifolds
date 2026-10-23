"""Single-device FP32 ANVIL + Adam: twin rails, six maps, lane equalization, cautious decay.

Ported from modded-nanoGPT record 92 (MIT). Dense Adam updates each step here;
geometry uses Adam, while both FFN projection pairs use the same ANVIL rule.
"""
import math
import torch

MAPS = [(3.923798038567, -6.095026865488, 3.905234618423),
        (3.278126713798, -3.328923386476, .989127286973),
        (3.505298394150, -5.137358782410, 1.968325560615),
        (2.815058591845, -3.685181239622, 1.417196497642),
        (2.245503932403, -2.443826979899, .963091710461),
        (2.256537145403, -2.166840097229, .929501253245)]


class Optimizer:
    def __init__(self, model, lr_scale, *, fuse_ks3_geometry=True):
        self.scale, self.matrices, self.adam = lr_scale, [], []
        for name, p in model.named_parameters():
            if name in ('qk_bank', 'vo_bank') or name.startswith('ff.') and name.endswith(('down.weight', 'up.weight')):
                transpose = name.endswith('up.weight')
                v = p.mT if transpose else p
                axis = -2 if v.shape[-2] >= v.shape[-1] else -1
                self.matrices.append(dict(p=p, transpose=transpose, fast=torch.zeros_like(v), slow=torch.zeros_like(v),
                                          energy=torch.zeros_like(v.mean(axis, keepdim=True)), axis=axis,
                                          multiplier=(2 if transpose else 1) * math.sqrt(max(1, v.shape[-2]/v.shape[-1]))))
            else:
                beta, mul, wd = (.9, .99), 1., .005
                if name == 'embed.weight' or name == 'lm_head.weight': beta, wd = (.5, .95), .75
                elif name == 'value_embeds': beta = (.0, .95)
                elif name == 'scalars' or name == 'resid_lambdas': mul, wd = 5., 0.
                elif name == 'post_lambdas': wd = 0.
                elif name == 'smear_gate.weight': mul, wd = .01, 0.
                elif 'mudd_gate' in name: mul, wd = .1, 0.
                elif 'mudd_' in name: mul, wd = .25, 0.
                elif name.startswith('ff.'): mul, wd = .125, 0.
                self.adam.append(dict(p=p, m=torch.zeros_like(p), v=torch.zeros_like(p), beta=beta, mul=mul, wd=wd))
        self.t = 0
        self.ks3_geometry = []
        if fuse_ks3_geometry:
            by_parameter = {id(s['p']): s for s in self.adam}
            for cell in model.ff.values():
                if hasattr(cell, '_ks3_nonlinear'):
                    self.ks3_geometry.append([by_parameter[id(p)] for p in
                                              (cell.coefficients, cell.raw_scale, cell.raw_shift)])

    def zero_grad(self):
        for s in self.matrices + self.adam:
            s['p'].grad = None

    @torch.no_grad()
    def step(self, progress, multiplier):
        self.t += 1
        warm, cool = 240/1194, 50/1194
        beta = .85 + .08 * min(1., progress/warm, max(0., (1-progress)/cool))
        fast_beta, mix = (.85, .4385) if progress >= 514/1194 else (beta, 1.)
        lr = .023 * self.scale * multiplier
        for s in self.matrices:
            p = s['p'].mT if s['transpose'] else s['p']
            g = s['p'].grad
            if g is None: continue
            g = g.mT if s['transpose'] else g
            s['fast'].lerp_(g, 1-fast_beta)
            s['slow'].lerp_(g, .02)
            x = g.lerp(mix*s['fast'] + (1-mix)*s['slow'], beta)
            x = x / (1.05*x.norm(dim=(-2, -1), keepdim=True) + 1e-6)
            tall = x.shape[-2] > x.shape[-1]
            for a, b, c in MAPS:
                gram = x.mT @ x if tall else x @ x.mT
                poly = b*gram + c*(gram @ gram)
                x = a*x + (x @ poly if tall else poly @ x)
            power = x.square().mean(s['axis'], keepdim=True)
            s['energy'].lerp_(power, .1)
            gain = s['energy'].clamp_min(1e-10).rsqrt()
            ratio = (power.sum((-2, -1), keepdim=True) / (power*gain.square()).sum((-2, -1), keepdim=True).clamp_min(1e-10)).sqrt()
            update = x*gain*ratio
            p.add_(p*(p*s['slow'] >= 0), alpha=-2.25*lr*lr*s['multiplier'])
            p.add_(update, alpha=-lr*s['multiplier'])
        lr = .008*self.scale*multiplier
        fused = set()
        if self.ks3_geometry:
            from experiments.common.ks3_int8 import geometry_adam_step
            for groups in self.ks3_geometry:
                if geometry_adam_step(groups, self.t, lr):
                    fused.update(id(s['p']) for s in groups)
        for s in self.adam:
            if id(s['p']) in fused: continue
            p, b = s['p'], s['beta']
            if p.grad is None: continue
            s['m'].lerp_(p.grad, 1-b[0])
            s['v'].lerp_(p.grad.square(), 1-b[1])
            update = s['m'] / (s['v'].sqrt() + 1e-10) * (math.sqrt(1-b[1]**self.t)/(1-b[0]**self.t))
            rate = lr*s['mul']
            p.add_(p*(p*update > 0), alpha=-rate*rate*s['wd'])
            p.add_(update, alpha=-rate)
