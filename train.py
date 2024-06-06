import os
import re
import time
import json
import copy
import psutil
import pickle
import argparse
import numpy as np
import torch
import wandb

import dnnlib
from dnnlib import EasyDict as edict
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc

from eval import evaluation_loop
from configs.config import config, update_config

def training_loop(
    run_dir,
    dataset_kwargs,
    dataloader_kwargs, 
    network_kwargs,
    diffusion_kwargs,
    loss_kwargs,
    optimizer_kwargs,
    augment_kwargs                   = None,      # Options for augmentation pipeline, None = disable.
    log_kwargs                       = None,
    seed                             = 0,         # Global random seed.
    batch_size                       = 2048,      # Total batch size for one training iteration.
    batch_gpu                        = None,      # Limit batch size per GPU, None = no limit.
    total_kobj                       = 50000,     # Training duration, measured in thousands of training puzzles.
    ema_halflife_kobj                = 500,       # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio                 = 0.05,      # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kobj                   = 10000,     # Learning rate ramp-up duration.
    lr_decay_kobj                    = 100000,    # Learning rate decaying interval.
    loss_scaling                     = 1,         # Loss scaling factor for reducing FP16 under/overflows.
    kobj_per_tick                    = 50,        # Interval of progress prints.
    snapshot_ticks                   = 50,        # How often to save network snapshots, None = disable.
    evaluate_ticks                   = 50,        # How often to evaluate network, None = disable.
    state_dump_ticks                 = 500,       # How often to dump training state, None = disable.
    resume_pkl                       = None,      # Start from the given network snapshot, None = random initialization.
    resume_state_dump                = None,      # Start from the given training state, None = reset training state.
    resume_kobj                      = 0,         # Start from the given training progress.
    cudnn_benchmark                  = True,      # Enable torch.backends.cudnn.benchmark?
    device                           = torch.device('cuda'),
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()
    
    # Load dataset.
    dist.print0('Loading dataset...')
    train_dataset = dnnlib.util.construct_class_by_name(split="train", **dataset_kwargs)
    train_sampler = misc.InfiniteSampler(dataset=train_dataset, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed, shuffle=True)
    train_dataloader = iter(torch.utils.data.DataLoader(dataset=train_dataset, sampler=train_sampler, batch_size=batch_gpu, **dataloader_kwargs))

    # Construct network.
    dist.print0('Constructing network...')
    interface_kwargs = dict(in_channels=train_dataset.num_channels, condition_channels=train_dataset.condition_dim, out_channels=train_dataset.num_channels)
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
    net.train().requires_grad_(True).to(device)

    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    diffuser = dnnlib.util.construct_class_by_name(**diffusion_kwargs) # training.loss.(VP|VE|EDM)Loss
    loss_fn = dnnlib.util.construct_class_by_name(diffuser=diffuser, **loss_kwargs) # training.loss.(VP|VE|EDM)Loss
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[dist.get_rank()], broadcast_buffers=False)
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        del data # conserve memory
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        optimizer.load_state_dict(data['optimizer_state'])
        del data # conserve memory
    
    # Set logger for logging loss and evaluation metrics
    if dist.get_rank() == 0:
        logger = dnnlib.util.construct_class_by_name(**log_kwargs)

    # Train.
    dist.print0(f'Training for {total_kobj} kpuzzles...')
    dist.print0()
    
    cur_nobj = resume_kobj * 1000
    cur_tick = 0
    tick_start_nobj = cur_nobj
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nobj // 1000, total_kobj)
    stats_jsonl = None
    while True:
        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                data = next(train_dataloader)
                data = {k: v.to(device)  for k, v in data.items()}
                loss = loss_fn(net=ddp, data=data, augment_pipe=augment_pipe)
                training_stats.report('Loss/loss', loss)
                loss.sum().mul(loss_scaling / batch_gpu_total).backward()

        # Update weights.
        for g in optimizer.param_groups:
            g['lr'] = optimizer_kwargs['lr'] * min(cur_nobj / max(lr_rampup_kobj * 1000, 1e-8), 1)
            g['lr'] = g['lr'] / (10 ** int(cur_nobj / max(lr_decay_kobj * 1000, 1e-8)))
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        optimizer.step()

        # Update EMA.
        ema_halflife_nobj = ema_halflife_kobj * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nobj = min(ema_halflife_nobj, cur_nobj * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nobj, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Perform maintenance tasks once per tick.
        cur_nobj += batch_size
        done = (cur_nobj >= total_kobj * 1000)
        if (not done) and (cur_tick != 0) and (cur_nobj < tick_start_nobj + kobj_per_tick * 1000):
            continue
        
        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kobj {training_stats.report0('Progress/kobj', cur_nobj / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kobj {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nobj - tick_start_nobj) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        # logging statistics per iteration
        if dist.get_rank() == 0 and (cur_nobj + batch_size) // 1000 != (cur_nobj // 1000):
            logger.log_iter(cur_nobj // 1000, loss, g['lr'])

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        # Evaluate network.
        if (evaluate_ticks is not None) and (done or cur_tick % evaluate_ticks == 0):
            if dist.get_rank() == 0:
                test_dataset = dnnlib.util.construct_class_by_name(split="test", **dataset_kwargs)
                test_dataloader = iter(torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_gpu, **dataloader_kwargs))
                evaluation_loop(
                        run_dir,
                        net,
                        test_dataloader,
                        diffuser,
                        logger,
                        cur_nobj
                )

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'network-snapshot-{cur_nobj//1000:06d}.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            del data # conserve memory

        # Save full dump of the training state.
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            torch.save(dict(net=net, optimizer_state=optimizer.state_dict()), os.path.join(run_dir, f'training-state-{cur_nobj//1000:06d}.pt'))

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()
        dist.update_progress(cur_nobj // 1000, total_kobj)

        # Update state.
        cur_tick += 1
        tick_start_nobj = cur_nobj
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------

def main():
    # project related args
    parse_args()
    cc = edict()
    cd, cn, cf, co, cl = config.dataset, config.network, config.diffusion, config.optimizer, config.loss

    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    cc.resume_pkl, cc.resume_state_dump = config.model_pkl, config.model_state
    cc.batch_size, cc.total_kobj = config.batch_size, config.total_kobj
    cc.ema_halflife_kobj, cc.ema_rampup_ratio = co.ema_halflife_kobj, co.ema_rampup_ratio
    cc.kobj_per_tick, cc.snapshot_ticks, cc.state_dump_ticks = config.kobj_per_tick, config.snapshot_ticks, config.state_dump_ticks
    cc.lr_decay_kobj, cc.lr_rampup_kobj = co.lr_decay_kobj, co.lr_rampup_kobj
    
    is_3d = ("3d" in config.exp_name)
    if "puzzlefusion" in config.exp_name:
        cc.dataset_kwargs = edict()
        if is_3d:
            cd.num_channels, cd.condition_dim = 9, 67
            cc.dataset_kwargs.class_name = 'data.CrosscutDataset3D'
            cc.dataset_kwargs.update(exp_name = config.exp_name, num_channels=cd.num_channels, condition_dim=cd.condition_dim)
        else:
            cd.num_channels, cd.condition_dim = 4, 66
            cc.dataset_kwargs.class_name = 'data.CrosscutDataset'
            cc.dataset_kwargs.update(exp_name = config.exp_name, num_channels=cd.num_channels, condition_dim=cd.condition_dim)
        cc.dataloader_kwargs = edict(pin_memory=True, num_workers=4, prefetch_factor=2)

        cc.network_kwargs = edict()
        cc.network_kwargs.class_name = 'network.transformer.TransformerModel'
        cc.network_kwargs.update(mid_channels=cn.mid_channels)

        cc.diffusion_kwargs = edict()
        cc.diffusion_kwargs.class_name = 'gaussian_diffusion.GaussianDiffusion'
        cc.diffusion_kwargs.update(num_timesteps=cf.num_timesteps, noise_schedule=cf.noise_schedule, predict_xstart=cf.predict_xstart)

        cc.loss_kwargs = edict()
        cc.loss_kwargs.class_name = 'loss.PFLoss'
        cc.loss_kwargs.update(is_3d = is_3d, use_matching_loss = cl.use_matching_loss)

    cc.optimizer_kwargs = edict()
    cc.optimizer_kwargs.class_name = 'torch.optim.AdamW'
    cc.optimizer_kwargs.update(lr=co.lr, weight_decay=co.weight_decay)

    # Description string.
    desc = f'{config.exp_name:s}-gpus{dist.get_world_size():d}-batch{cc.batch_size:d}'
        
    # Pick output directory.
    if dist.get_rank() != 0:
        cc.run_dir = None
    elif config.nosubdir:
        cc.run_dir = config.outdir
    else:
        prev_run_dirs = []
        if os.path.isdir(config.outdir):
            prev_run_dirs = [x for x in os.listdir(config.outdir) if os.path.isdir(os.path.join(config.outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        cc.run_dir = os.path.join(config.outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(cc.run_dir)
        
        # logger related args
        cc.log_kwargs = edict()
        cc.log_kwargs.class_name = 'log.Logger'
        cc.log_kwargs.update(project_name = config.project, exp_name = config.exp_name, tags = f'{cur_run_id:05d}-{desc}', run_dir=cc.run_dir)

    # Dry run?
    if config.dry_run:
        dist.print0('Dry run; exiting.')
        return

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(cc.run_dir, exist_ok=True)
        with open(os.path.join(cc.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(cc, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(cc.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    training_loop(**cc) 
    wandb.finish()

def parse_args():
    parser = argparse.ArgumentParser(description='Equivariant Learning to Shape Puzzles')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)
    args = parser.parse_args()
    update_config(args.cfg)

if __name__ == '__main__':
    main()