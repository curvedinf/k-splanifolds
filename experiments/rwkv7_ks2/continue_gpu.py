"""Resume a saved GPU optimizer at constant terminal LR, with persistent graphs.

An optional command file lets an external coordinator advance all seeds in
lockstep. Checkpoints are written before the matching atomic results update.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F
import run
from gpu_recurrence import recurrence


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


class Continuation:
    def __init__(self, resume, data, out, lr=.0002, verify=False):
        self.out, self.lr = out, lr
        self.wall_begin = time.perf_counter()
        out.mkdir(parents=True, exist_ok=True)
        checkpoint = torch.load(resume, map_location='cpu', weights_only=True)
        self.source = checkpoint.get('run_result') or json.loads(resume.with_name('results.json').read_text())
        self.config = dict(self.source['arguments'])
        self.step = int(checkpoint['step'])
        self.start_step = self.step
        self.source_path = str(resume)
        assert self.step == self.config['steps']
        self.train, val, self.manifest = run.prepare_data(data, data)
        assert self.manifest == self.source['data']
        run.recurrence = recurrence
        c = self.config
        self.model = run.Model(c['seed'], c['variant'], width=c['width'], layers=c['layers'],
                               ffn_charts=c['ffn_charts'], fused_cpu=False).cuda()
        self.model.load_state_dict(checkpoint['model'])
        base = run.make_optimizer(self.model, c['control_update_scale'], c['ffn_charts'])
        groups = [dict(params=g['params'], lr=torch.tensor(g['lr'], device='cuda'),
                       lr_scale=g['lr_scale'], weight_decay=g['weight_decay']) for g in base.param_groups]
        self.optimizer = torch.optim.AdamW(groups, capturable=True, foreach=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        for group in self.optimizer.param_groups:
            group['lr'] = torch.as_tensor(group['lr'], device='cuda').clone()
            group['lr'].fill_(lr * group['lr_scale'])
        steps = torch.stack([s['step'] for s in self.optimizer.state.values()])
        assert steps.unique().tolist() == [float(self.step)]
        saved_sampler = checkpoint.get('sampler_state')
        del base, checkpoint
        self.generator = torch.Generator().manual_seed(c['seed'] + 10000)
        # Advance exactly the same CPU sampler used by the original fixed runs.
        torch.randint(len(self.train) - 65, (self.step, 4), generator=self.generator)
        if saved_sampler is not None:
            assert torch.equal(saved_sampler, self.generator.get_state())
        state = self.generator.get_state()
        self.batch = self.next_batches(1)[0].clone()
        self.generator.set_state(state)
        self.invalid = torch.zeros((), dtype=torch.int64, device='cuda')
        weights = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        moments = {p: {k: v.detach().clone() for k, v in s.items()}
                   for p, s in self.optimizer.state.items()}

        def restore():
            self.model.load_state_dict(weights)
            for p, saved in moments.items():
                for key, value in saved.items():
                    self.optimizer.state[p][key].copy_(value)
            self.optimizer.zero_grad(set_to_none=False)
            self.invalid.zero_()
            torch.cuda.synchronize()

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.update()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        restore()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.loss, self.norm = self.update()
        restore()
        if verify:
            self.update()
            expected = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            expected_moments = {p: {k: v.detach().clone() for k, v in s.items()}
                                for p, s in self.optimizer.state.items()}
            restore()
            self.graph.replay()
            torch.cuda.synchronize()
            for name, value in self.model.state_dict().items():
                torch.testing.assert_close(value, expected[name], atol=2e-6, rtol=2e-5)
            for p, saved in expected_moments.items():
                for key, value in saved.items():
                    torch.testing.assert_close(self.optimizer.state[p][key], value, atol=2e-6, rtol=2e-5)
            restore()
            print('PASS: resumed eager/graph update, including all Adam states; capture updates restored', flush=True)
        del weights, moments
        self.eval_batches = {}
        for name, array, seed, count in [('val_loss', val, 12345, 32),
                                         ('monitor_loss', val, 54321, 256),
                                         ('sampled_train_loss', self.train, 98765, 256)]:
            positions = torch.randint(len(array) - 65, (count, 4),
                                      generator=torch.Generator().manual_seed(seed))
            self.eval_batches[name] = array[positions[:, :, None] + torch.arange(65)].cuda()
        self.eval_input = self.eval_batches['val_loss'][0].clone()
        with torch.no_grad():
            self.model(self.eval_input[:, :-1])
            self.eval_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.eval_graph):
                self.eval_loss = F.cross_entropy(self.model(self.eval_input[:, :-1]).flatten(0, 1),
                                                 self.eval_input[:, 1:].flatten())
        self.history = list(self.source['history'])
        self.training_seconds = self.source['training_seconds']
        measured = self.evaluate()
        assert abs(measured['val_loss'] - self.history[-1]['val_loss']) < 2e-6
        self.history[-1] = dict(self.history[-1], **measured)
        self.save()

    def next_batches(self, count):
        positions = torch.randint(len(self.train) - 65, (count, 4), generator=self.generator)
        return self.train[positions[:, :, None] + torch.arange(65)].cuda()

    def update(self):
        self.optimizer.zero_grad(set_to_none=False)
        loss = F.cross_entropy(self.model(self.batch[:, :-1]).flatten(0, 1), self.batch[:, 1:].flatten())
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., foreach=True)
        self.invalid.add_((~(torch.isfinite(loss) & torch.isfinite(norm))).long())
        self.optimizer.step()
        return loss, norm

    def evaluate(self):
        measured = {}
        for name, batches in self.eval_batches.items():
            losses = []
            for batch in batches:
                self.eval_input.copy_(batch)
                self.eval_graph.replay()
                losses.append(self.eval_loss.clone())
            measured[name] = torch.stack(losses).mean().item()
        return measured

    def advance(self, target):
        while self.step < target:
            count = min(5000, target - self.step)
            batches = self.next_batches(count)
            torch.cuda.synchronize()
            begin = time.perf_counter()
            for batch in batches:
                self.batch.copy_(batch)
                self.graph.replay()
            torch.cuda.synchronize()
            self.training_seconds += time.perf_counter() - begin
            self.step += count
            if self.invalid.item():
                raise RuntimeError('Nonfinite loss or gradient in continuation')
            entry = dict(step=self.step, train_loss=self.loss.item(), gradient_norm=self.norm.item(),
                         training_seconds=self.training_seconds, base_lr=self.lr, **self.evaluate())
            self.history.append(entry)
            self.save()
            print(json.dumps(entry), flush=True)

    def save(self):
        result = dict(self.source)
        result.update(arguments=dict(self.config, out=str(self.out), steps=self.step),
                      history=self.history, training_seconds=self.training_seconds,
                      wall_seconds=None,
                      invalid_steps=int(self.invalid.item()), training_targets=self.step * 256,
                      continuation=dict(source_checkpoint=self.source_path, start_step=self.start_step,
                                        base_lr=self.lr, sampler='Original CPU generator continued without reset',
                                        training_seconds=self.training_seconds - self.source['training_seconds'],
                                        wall_seconds=time.perf_counter() - self.wall_begin,
                                        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        temporary = self.out / 'checkpoint.pt.tmp'
        torch.save(dict(model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                        step=self.step, sampler_state=self.generator.get_state(), run_result=result), temporary)
        temporary.replace(self.out / 'checkpoint.pt')
        atomic_json(self.out / 'results.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--lr', type=float, default=.0002)
    parser.add_argument('--verify-resume', action='store_true')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--steps', type=int)
    mode.add_argument('--command', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    worker = Continuation(args.resume, args.data, args.out, args.lr, args.verify_resume)
    if args.steps is not None:
        if args.steps < worker.step:
            raise ValueError('Target steps precede checkpoint')
        worker.advance(args.steps)
        return
    while True:
        command = json.loads(args.command.read_text())
        if command.get('stop'):
            return
        target = command['target_step']
        if target > worker.step:
            worker.advance(target)
        else:
            time.sleep(.2)


if __name__ == '__main__':
    main()
