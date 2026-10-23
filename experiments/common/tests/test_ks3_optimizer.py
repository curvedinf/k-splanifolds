"""Fused geometry updates against the unchanged nanoGPT Adam equations."""
import copy
import math
import unittest

import torch

from experiments.common.ks3_int8 import geometry_adam_step, pack_weight, quantize


@unittest.skipUnless(torch.cuda.is_available(), 'GPU and Triton required')
class GeometryOptimizer(unittest.TestCase):
    def test_quantization_matches_cpu_reference(self):
        torch.manual_seed(117)
        for width in (1,6,31,32,33,96,337):
            values=torch.randn(17,width)*torch.logspace(-9,8,17)[:,None]
            reference=pack_weight(values)
            actual=quantize(values.cuda())
            for a,b in zip(actual,reference):
                torch.testing.assert_close(a.cpu(),b,atol=0,rtol=0)
        # Exact half-way ties with a representable FP16 group scale.
        values=torch.tensor([.5,1.5,2.5,126.5,127.,-.5,-1.5,-2.5,-126.5,-127.])[None,:]*.125
        for a,b in zip(quantize(values.cuda()),pack_weight(values)):
            torch.testing.assert_close(a.cpu(),b,atol=0,rtol=0)

    def test_adam_parameters_and_moments(self):
        torch.manual_seed(91)
        torch.set_num_threads(2)
        for rank,decay in ((1,0.),(3,.005),(16,0.)):
            groups=[]
            for shape in ((7,rank,6),(7,),(7,)):
                p=torch.nn.Parameter(torch.randn(shape,device='cuda')*.1)
                groups.append(dict(p=p,m=torch.zeros_like(p),v=torch.zeros_like(p),
                                   beta=(.9,.99),mul=.125,wd=decay))
            expected=copy.deepcopy(groups)
            for step in range(1,51):
                lr=.002*(1-step/100)
                for a,b in zip(groups,expected):
                    a['p'].grad=torch.randn_like(a['p'])*(.2 if step%2 else 3.)
                    b['p'].grad=a['p'].grad.clone()
                self.assertTrue(geometry_adam_step(groups,step,lr))
                with torch.no_grad():
                    for s in expected:
                        p,b=s['p'],s['beta']
                        s['m'].lerp_(p.grad,1-b[0])
                        s['v'].lerp_(p.grad.square(),1-b[1])
                        update=s['m']/(s['v'].sqrt()+1e-10)*(math.sqrt(1-b[1]**step)/(1-b[0]**step))
                        rate=lr*s['mul']
                        p.add_(p*(p*update>0),alpha=-rate*rate*s['wd'])
                        p.add_(update,alpha=-rate)
                for a,b in zip(groups,expected):
                    for name in ('p','m','v'):
                        torch.testing.assert_close(a[name],b[name],atol=2e-7,rtol=2e-6)
            groups[0]['p'].grad=None
            before=[g['p'].detach().clone() for g in groups]
            self.assertFalse(geometry_adam_step(groups,51,.002))
            for g,p in zip(groups,before):
                torch.testing.assert_close(g['p'],p,atol=0,rtol=0)


if __name__=='__main__':
    unittest.main()
