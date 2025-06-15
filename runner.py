
import collections
import dataclasses
import datetime
import math
import os
import pathlib
import time
from typing import Optional
import random
import sys

import torch

import configuration
import data as draftdata
import network

class Runner:
    def __init__(self, config: configuration.Config_train):
        self.config = config

    def init(self):
        os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:20"
        torch.set_printoptions(linewidth=120, sci_mode=False)

        self.model_dir = pathlib.Path('models')
        self.start_time = datetime.datetime.now().isoformat().replace('T', '_').replace(':', '-')[:19]

        self.ds_meta = draftdata.load_datasets_meta(self.config)

    def print(self, *x, **y):
        if self.config.silent: return
        print(*x, **y)
        
    def load_meta(self):
        if self.config.meta_cardinfo:
            m = max(map(len, self.tokens_meta.values()))
            self.meta_cardinfo = torch.zeros(self.config.vocab_size, m, dtype=torch.int64)
            for i, lst in self.tokens_meta.items():
                self.meta_cardinfo[i,:len(lst)] = torch.tensor(lst, dtype=torch.int64)
        else:
            self.meta_cardinfo = None
                
        if self.config.meta_pickinfo:
            P = 15
            m = max(P - i.set_packsize for i in self.ds_meta.data) + 1
            self.meta_pickinfo = torch.zeros(m, self.config.seq_cards, 2, dtype=torch.int64)
            for i in range(m):
                for pack in range(3):
                    for pick in range(P-i):
                        it = pack*(P-i) + pick
                        self.meta_pickinfo[i,it,0] = self.tokens[f'%pack={pack}']
                        self.meta_pickinfo[i,it,1] = self.tokens[f'%pick={pick}']
        else:
            self.meta_pickinfo = None

    def _filter_hq(self, ds):
        idx = configuration.Token_type.LEAGUE.header_index
        toks = [self.tokens.index[i] for i in ('$diamond', '$mythic')]
        indices = [it for it,(picks,_) in enumerate(ds) if picks[idx] in toks]
        return torch.utils.data.Subset(ds, indices)
            
    def load_dataset(self, datasets_json=None):
        if self.config.testdata_only:
            self.dataset_test = torch.utils.data.TensorDataset(*torch.load(self.config.testdata_only_path))
            self.print(f'Testdata loaded (test={len(self.dataset_test)})')
        else:
            dataset = draftdata.load_dataset(self.ds_meta.data, self.config, datasets_json)
            generator_split = torch.Generator().manual_seed(0)
            n = self.config.testdata_size
            m = self.config.testdata_size_max - n + self.config.traindata_skip
            self.dataset_test, _, self.dataset_train = torch.utils.data.random_split(dataset, [n, m, len(dataset)-n-m], generator_split)

            self.total_batches = math.ceil(len(self.dataset_train) // self.config.batch_size) * self.config.epochs
            self.print(f'Dataset loaded (train={len(self.dataset_train)}, test={len(self.dataset_test)})')

    def load_loader(self):
        if self.config.filter_hq:
            n, m = len(self.dataset_train), len(self.dataset_test)
            self.dataset_train = self._filter_hq(self.dataset_train)
            self.dataset_test = self._filter_hq(self.dataset_test)
            nn, mm = len(self.dataset_train), len(self.dataset_test)
            self.print(f'Filtering datasets (train: {n} -> {nn}, test {m} -> {mm})')

        generator_loader = torch.Generator()
        self.print(f'Seed for training loader: {generator_loader.seed()}')
        if not self.config.testdata_only:
            self.loader_train = torch.utils.data.DataLoader(self.dataset_train, batch_size=self.config.batch_size, shuffle=True, pin_memory=True, generator=generator_loader)
        self.loader_test = torch.utils.data.DataLoader(self.dataset_test, batch_size=self.config.batch_size, shuffle=False, pin_memory=True)

    def _get_newest_checkpoint(self):
        checkpoints = [i for i in self.model_dir.iterdir() if i.suffix == '.pt']
        if not checkpoints:
            print('Error: could not find any existing checkpoints to load')
        checkpoints.sort()
        return checkpoints[-1]

    def load_model(self):
        if self.config.device:
            self.device = self.config.device
        else:
            self.device = torch.accelerator.current_accelerator()
            if self.device is None:
                self.device = 'cpu'
            self.print('Using device', self.device)
            
        if not self.config.get_checkpoint_load():
            if self.config.meta_any():
                self.tokens = draftdata.Tokens.create(self.ds_meta.tokens_base.token_list, self.ds_meta.tokens_meta)
            else:
                self.tokens = self.ds_meta.tokens_base
            checkpoint = None
        else:
            if self.config.checkpoint_load_path:
                checkpoint_path = pathlib.Path(self.config.checkpoint_load_path)
            else:
                checkpoint_path = self._get_newest_checkpoint()

            checkpoint = torch.load(checkpoint_path, map_location=torch.device(self.device))
            self.print(f'Loaded checkpoint {checkpoint_path}')
            self.tokens = draftdata.Tokens(checkpoint['token_list'], checkpoint.get('tokens_meta'))

            # Revert all custom model configuration and load the values from the checkpoint
            config_default = configuration.Config()
            self.config.merge({i: getattr(config_default, i) for i in configuration.Config_model.keys()})
            self.config.merge(checkpoint.get('config', {}))
            self.config.merge({i: self.config._args_config[i] for i in configuration.Config_model.keys() if i in self.config._args_config})

        self.config.vocab_size = len(self.tokens)
        self.model = network.Decoder(self.config)

        self.load_meta()
        
        if checkpoint is not None:
            self.model.load_state_dict(checkpoint['model'], strict=self.config.checkpoint_load_strict)
            
        self.model.to(self.device)
        self.model.train(self.config.training)

        if self.config.training:
            import torch.optim.lr_scheduler as lrs

            if self.config.schedule_free:
                import schedulefree
                self.optimizer = schedulefree.AdamWScheduleFree(
                    self.model.parameters(),
                    lr=self.config.learning_rate,
                    betas=[self.config.adam_beta1, self.config.adam_beta2],
                    eps=self.config.adam_eps,
                    warmup_steps=self.config.lr_warmup,
                )
            else:
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.config.learning_rate,
                    betas=[self.config.adam_beta1, self.config.adam_beta2],
                    eps = self.config.adam_eps,
                )
                
            if self.config.tensorboard:
                from torch.utils.tensorboard import SummaryWriter        
                self.tensorboard_writer = SummaryWriter()

            if checkpoint and self.config.train_continue:
                optim_path = checkpoint_path.with_suffix('.optim')
                if not optim_path.exists():
                    self.print('Warning: no optimiser data found, training will restart')
                else:
                    optim = torch.load(optim_path)
                    self.print(f'Loaded optimizer checkpoint {optim_path}')

                    self.optimizer.load_state_dict(optim['optimizer'])
                    #self.scheduler.load_state_dict(optim['scheduler'])

                    for g in self.optimizer.param_groups:
                        g['lr'] = self.config.learning_rate
                        
            if self.config.schedule_free:
                self.scheduler = lrs.ConstantLR(self.optimizer, factor=1)
            else:
                self.scheduler = lrs.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.total_batches,
                    eta_min=self.config.learning_rate_min,
                )

                if self.config.lr_warmup:
                    warmup_scheduler = lrs.LinearLR(self.optimizer, start_factor=0.01, total_iters=self.config.lr_warmup)
                    self.scheduler = lrs.ChainedScheduler([warmup_scheduler, self.scheduler], optimizer=self.optimizer)

        else:
            self.optimizer = None
            self.scheduler = None
            

        self.print('Model loaded.')
        for name, module in self.model.named_children():
            params = sum(p.numel() for p in module.parameters())
            self.print(f"  {name}: {params:,} parameters")


    def init_and_load(self):
        self.init()
        self.load_dataset()
        self.load_model()
        self.load_loader()

    def save_model(self):
        self.model.eval()
        if hasattr(self.optimizer, 'eval'):
            self.optimizer.eval()
        
        self.model_dir.mkdir(exist_ok=True)
        
        path = self.model_dir / f'model_{self.start_time}.pt'
        path_optim = path.with_suffix('.optim')
        temp = lambda x: x.with_name('~' + x.name)

        save_config = {i: getattr(self.config, i) for i in configuration.Config_model.keys()}
        data = {
            'model': self.model.state_dict(),
            'batch': self.cur_batch,
            'token_list': self.tokens.token_list,
            'config': save_config,
        }
        if self.tokens.tokens_meta:
            data['token_meta'] = self.tokens.tokens_meta
        torch.save(data, temp(path))
            
        if self.config.checkpoint_save_optim:
            data_optim = {
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
            }
            torch.save(data_optim, temp(path_optim))
            
        os.replace(temp(path), path)
        if self.config.checkpoint_save_optim:
            os.replace(temp(path_optim), path_optim)

    def compute_logits(self, picks, packs, skip_packs=False, mask_packs=True):
        device = next(self.model.parameters()).device
        if self.config.meta_cardinfo: # TODO look at whether this makes sense (cardinfo, not pickinfo?)
            pad = (picks == 0).sum(1) // 3
            position_meta = self.meta_pickinfo[pad].to(device)
            self.meta_cardinfo = self.meta_cardinfo.to(device)
        else:
            position_meta = None

        picks = picks.to(device, dtype=torch.int64)
        packs = packs.to(device, dtype=torch.int64)

        logits, logits_rate = self.model(picks, packs=packs, token_meta=self.meta_cardinfo, position_meta=position_meta, skip_packs=skip_packs, mask_packs=mask_packs)
        return logits, logits_rate
        
    def compute_loss(self, picks, packs, rates, skip_packs=False):
        device = next(self.model.parameters()).device
        picks = picks.to(device, dtype=torch.int64)
        rates = rates.to(device, dtype=torch.float) / 255.0

        if self.config.debug_images:
            import debug
        else:
            from debug import Dummy as debug
        debug.imgdump(picks, 'picks')
        debug.imgdump(packs, 'packs')
        debug.imgdump(rates, 'rates')
        
        logits, logits_rate = self.compute_logits(picks, packs, skip_packs=skip_packs)

        debug.imgdump(logits, 'logits')
        debug.imgdump(logits_rate, 'logits_rate')
        if self.config.debug_images: sys.exit(0)
        
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_picks  = picks[:,1:].reshape(-1)

        loss = {}
        loss['picks'] = torch.nn.functional.cross_entropy(flat_logits, flat_picks, ignore_index=0)
        loss['total'] = loss['picks'].clone()

        if self.config.do_rates:
            rates_idx = torch.tril_indices(self.config.seq_cards, self.config.seq_cards, device=device)
            rates_seq = rates[:,rates_idx[1]]
            loss['rates'] = torch.nn.functional.binary_cross_entropy_with_logits(
                input=logits_rate,
                target=rates_seq,
                weight=((rates_idx[0]+1)/self.config.seq_cards),
                pos_weight=torch.tensor([self.config.rate_loss_posw], device=device)
            )
            loss['total'] += loss['rates'] * self.config.rate_loss_fac

        loss['accuracy'] = torch.mean(torch.argmax(logits[:,self.config.seq_header-1:,:], dim=2) == picks[:,self.config.seq_header:], dtype=torch.float32)
        
        return loss

    def compute_test_loss(self, skip_packs=False):
        self.model.eval()
        if hasattr(self.optimizer, 'eval'):
            self.optimizer.eval()
        
        with torch.no_grad():
            loss = collections.defaultdict(float)
            loss_count = 0
            for tpicks, tpacks, trates in self.loader_test:
                ld = self.compute_loss(tpicks, tpacks, trates, skip_packs=skip_packs)
                for k, val in ld.items():
                    loss[k] += val.item()
                loss_count += 1

            loss = {k: val/loss_count for k,val in loss.items()}
            
        return loss

    def train_step(self, picks, packs, rates):
        self.model.train()
        if hasattr(self.optimizer, 'train'):
            self.optimizer.train()
        
        skip_packs = self.cur_batch > 0 and random.random() <= self.config.pack_skip_frac
        
        loss = self.compute_loss(picks, packs, rates, skip_packs=skip_packs)

        self.print_loss.append(loss['total'].item())
        if self.config.tensorboard:
            for k, v in loss.items():
                self.tensorboard_writer.add_scalar('Loss/train/'+k, v.item(), self.cur_batch)
            
        loss['total'].backward()

        total_norm = torch.nn.utils.get_total_norm([i.grad for i in self.model.parameters() if i.grad is not None])
        if self.config.tensorboard:
            self.tensorboard_writer.add_scalar('Loss/train/gradient', total_norm.item(), self.cur_batch)
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grads_with_norm_(self.model.parameters(), max_norm=self.config.grad_clip, total_norm=total_norm)

        self.optimizer.step()
        self.optimizer.zero_grad()

        lr, = self.scheduler.get_last_lr()
        if self.config.tensorboard:
            self.tensorboard_writer.add_scalar('Info/rate', lr, self.cur_batch)
        self.scheduler.step()

        if self.config.period_test >= 0 and self.cur_batch >= self.test_batch + self.config.period_test:
            self.test_batch = self.cur_batch
            
            test_loss = self.compute_test_loss()
            
            if self.config.tensorboard:
                for k, v in test_loss.items():
                    self.tensorboard_writer.add_scalar('Loss/test/'+k, v, self.cur_batch)
                
            self.print_messages['1_test_loss'] = f'test_loss: {test_loss["total"]:>7f}'

        now = time.time()
        if now >= self.save_time + self.config.period_save:
            if self.config.checkpoint_save:
                self.save_model()
                self.print_messages['2_saved'] = 'saved'
            self.save_time = now

            if self.config.tensorboard:
                if self.config.tensorboard_embedding:
                    self.tensorboard_writer.add_embedding(
                        self.model.embedding.weight,
                        metadata=self.tokens_list,
                        global_step=self.cur_batch
                    )
        
        if now >= self.print_time + self.config.period_print:
            speed = self.config.batch_size * (self.cur_batch+1 - self.print_batch) / (now - self.print_time)
            self.print_time = now
            self.print_batch = self.cur_batch
                
            ploss = sum(self.print_loss) / len(self.print_loss)
            self.print_loss = []

            current = self.cur_batch + 1
            w = len(str(self.total_batches))
            self.print_messages['0'] = f'[{current:>{w}d}/{self.total_batches:>{w}d}]  loss: {ploss:>7f}  {speed:.0f} inst/s'
            
            lst = list(self.print_messages.items())
            lst.sort()
            self.print('  '.join(i for _,i in lst))
            self.print_messages = {}

            if self.config.tensorboard:
                self.tensorboard_writer.flush()

        if now >= self.train_start + self.config.train_duration_max:
            return True

        if self.cur_batch * self.config.batch_size >= self.config.train_instances_max >= 0:
            return True

        for tmin, lmax in self.config.train_cutoffs:
            if now >= self.train_start + tmin and loss['total'].item() > lmax:
                return True

        return False

    def train(self):
        self.train_start = time.time()
        self.print_time = self.train_start
        self.print_batch = 0
        self.print_loss = []
        self.print_messages = {}
        self.save_time = 0.0
        self.test_batch = 0

        
        if self.config.debug_profile:
            from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

            if self.config.tensorboard:
                trace_handler = tensorboard_trace_handler(self.tensorboard_writer.log_dir)
            else:
                def trace_handler(p):
                    output = p.key_averages().table(sort_by='cuda_time_total', row_limit=10)
                    print(output)
                    os.makedirs('traces', exist_ok=True)
                    p.export_chrome_trace("traces/trace_" + str(p.step_num) + ".json")

            prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                schedule=torch.profiler.schedule(
                    wait=0, warmup=4, active=10, repeat=1,
                ),
                on_trace_ready=trace_handler
            )
            prof.__enter__()
        else:
            prof = None
        
        stop = False
        for epoch in range(self.config.epochs):
            self.cur_epoch = epoch
            for epoch_batch, (picks, packs, rates) in enumerate(self.loader_train):
                self.cur_batch = epoch * len(self.loader_train) + epoch_batch
                stop = self.train_step(picks, packs, rates)
                if prof: prof.step()
                if stop: break
            if stop: break
                
            self.print(f'Finished epoch {epoch+1}/{self.config.epochs}')

        self.save_model()
        
        if prof: prof.__exit__()
