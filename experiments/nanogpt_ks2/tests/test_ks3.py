"""KS3 nanoGPT control, optimizer, causal training and checkpoint integration."""
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.nanogpt_ks2.model import GPT
from experiments.nanogpt_ks2.optim import Optimizer


class ControlCompatibility(unittest.TestCase):
    def test_torch_control_initialization_and_budget(self):
        torch.set_num_threads(2)
        torch.manual_seed(11)
        mlp=GPT('mlp',vocab=128,ngram_rows=64)
        torch.manual_seed(11)
        ks3=GPT('ks3',vocab=128,ngram_rows=64,ks3_backend='torch')
        for name,value in mlp.state_dict().items():
            if not name.startswith('ff.'):
                self.assertTrue(torch.equal(value,ks3.state_dict()[name]),name)
        self.assertEqual(ks3.config['ks3_subspaces'],337)
        self.assertEqual(sum(p.numel() for p in ks3.ff['0'].parameters()),67400)
        self.assertEqual(sum(p.numel() for p in mlp.ff['0'].parameters()),67584)
        tokens=torch.randint(0,128,(16,));docs=torch.zeros_like(tokens)
        torch.testing.assert_close(mlp(tokens,16,32,docs),ks3(tokens,16,32,docs),atol=0,rtol=0)


@unittest.skipUnless(torch.cuda.is_available(),'GPU and Triton required')
class FusedIntegration(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False

    def model(self,rank=1):
        return GPT('ks3',vocab=128,ngram_rows=64,ks3_rank=rank).cuda()

    def test_optimizer_ownership_training_and_causality(self):
        model=self.model()
        optimizer=Optimizer(model,.25)
        matrix_ids={id(group['p']) for group in optimizer.matrices}
        adam_ids={id(group['p']) for group in optimizer.adam}
        for cell in model.ff.values():
            self.assertIn(id(cell.down.weight),matrix_ids)
            self.assertIn(id(cell.up.weight),matrix_ids)
            for p in (cell.coefficients,cell.raw_scale,cell.raw_shift):
                self.assertIn(id(p),adam_ids)
        up=next(group for group in optimizer.matrices if group['p'] is model.ff['0'].up.weight)
        self.assertTrue(up['transpose'])
        tokens=torch.randint(0,128,(17,),device='cuda')
        docs=torch.zeros_like(tokens[:-1])
        initial=model.ff['0'].coefficients.detach().clone()
        for step in range(3):
            optimizer.zero_grad()
            hidden=model(tokens[:-1],16,32,docs)
            loss=F.cross_entropy(F.linear(hidden,model.head_weight()),tokens[1:])
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            optimizer.step(step/100,1.)
        self.assertFalse(torch.equal(initial,model.ff['0'].coefficients))
        altered=tokens[:-1].clone();altered[8:]=torch.randint(0,128,(8,),device='cuda')
        with torch.no_grad():
            torch.testing.assert_close(model(tokens[:-1],16,32,docs)[:8],model(altered,16,32,docs)[:8],atol=2e-5,rtol=2e-5)

    def test_checkpoint_reload(self):
        from experiments.nanogpt_ks2.bench_forward import load_checkpoint
        model=self.model(rank=3)
        for cell in model.ff.values():
            torch.nn.init.normal_(cell.up.weight,std=.005)
        tokens=torch.randint(0,128,(16,),device='cuda');docs=torch.zeros_like(tokens)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'ks3_11.pt'
            torch.save(dict(kind='ks3',model_config=model.config,model=model.state_dict(),
                            args=dict(width=96,rank=8),history=[dict(step=0)]),path)
            restored,_=load_checkpoint(path,device='cuda')
            self.assertEqual(set(model.state_dict()),set(restored.state_dict()))
            self.assertFalse(any('_ks3_tanh_lut' in name for name in restored.state_dict()))
            with torch.no_grad():
                torch.testing.assert_close(model(tokens,16,32,docs),restored(tokens,16,32,docs),atol=0,rtol=0)


if __name__=='__main__':
    unittest.main()
