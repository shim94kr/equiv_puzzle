
import re
import os
import time
import torch
import pickle
import argparse

import dnnlib
from torch_utils import misc
from torch_utils import distributed as dist
from dnnlib import EasyDict as edict
from configs.config import config, update_config
from metric import transform_pieces, get_pieces, get_pieces_3d, compute_metric
from visualize import draw_pieces, draw_pieces_3d
from torch_utils import eval_stats


def evaluation_loop(run_dir,
                    net,
                    dataloader,
                    diffuser,
                    logger,
                    cur_nobj = 0,
                    num_timesteps = 1000,
                    clip_denoised = True,
                    device = torch.device('cuda'),
):
    # Initialize.
    start_time = time.time()
    timesteps = list(range(num_timesteps))[::-1]
    nb_offset = 0

    for data in dataloader:
        data = {k: v.to(device)  for k, v in data.items()}
        x_gt = torch.cat([data["t"], data["rot"]], dim=-1)
        x_t = torch.randn_like(x_gt)
        x_shape = x_gt.shape

        model_kwargs = {"others": data}
        samples = [x_t]

        for t in timesteps:
            if t % 50 == 0:
                samples.append(x_t)
            t_ = torch.tensor([t] * x_shape[0], device=device)
            with torch.no_grad():
                x_t = diffuser.p_sample(
                    net,
                    x_t,
                    t_,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )['sample']
        
        samples = torch.stack(samples, dim=1)
        sample_gt = x_gt.unsqueeze(1)
        is_3d = "3d" in run_dir
        gt_piece = transform_pieces(data["vertices"], sample_gt, data["piece_mask"], is_3d = is_3d)
        pred_pieces = transform_pieces(data["vertices"], samples, data["piece_mask"], is_3d = is_3d)
        if "3d" in run_dir:
            gt_pieces = get_pieces_3d(gt_piece[:, -1:], data["faces_mask"], data["faces_piece_idx"], data["faces_padding_mask"], data["piece_idx"], data["padding_mask"])
            pred_pieces = get_pieces_3d(pred_pieces[:, -1:], data["faces_mask"], data["faces_piece_idx"], data["faces_padding_mask"], data["piece_idx"], data["padding_mask"])
            draw_pieces_fn = draw_pieces_3d
        else:
            gt_pieces = get_pieces(gt_pieces[:, -1:], data["piece_idx"], data["padding_mask"])
            pred_pieces = get_pieces(pred_pieces[:, -1:], data["piece_idx"], data["padding_mask"])
            draw_pieces_fn = draw_pieces
        gt_pieces_for_metric = [g[-1] for g in gt_pieces]
        pred_pieces_for_metric = [p[-1] for p in pred_pieces]
        metric = compute_metric(gt_pieces_for_metric, pred_pieces_for_metric, is_3d = is_3d)
        #draw_pieces_fn(run_dir, cur_nobj, 'gt', gt_pieces, nb_offset=nb_offset)
        draw_pieces_fn(run_dir, cur_nobj, 'pred', pred_pieces, nb_offset=nb_offset)

        nb_offset+=sample_gt.shape[0]

        for k, v in metric.items():
            eval_stats.report0(f'Eval/{k}', v)
            
    end_time = time.time()
    eval_stats.default_collector.update()
    dict_all = eval_stats.default_collector.as_dict()
    fields = [f"Eval time {dnnlib.util.format_time(end_time - start_time):<12s}"]
    for k, v in dict_all.items():
        if 'Eval/' in k:
            fields += [f"{k} {v['mean']:<6.2f}"]
            if dist.get_rank() == 0:
                logger.log_eval(cur_nobj // 1000, k, v['mean'])
    torch.cuda.reset_peak_memory_stats()
    dist.print0(' '.join(fields))
    eval_stats.default_collector.reset()

def main():
    # project related args
    parse_args()
    dist.init()
    cc = edict()
    cd, cf, cn = config.dataset, config.diffusion, config.network
    
    batch_size = config.batch_size
    network_pkl, model_state = config.model_pkl, config.model_state
    if "puzzlefusion" in config.exp_name:
        dataset_kwargs = edict()
        if "3d" in config.exp_name:
            dataset_kwargs.class_name = 'data.CrosscutDataset3D'
            cd.num_channels, cd.condition_dim = 9, 67
            dataset_kwargs.update(project = config.project, num_channels=cd.num_channels, condition_dim=cd.condition_dim)
        else:
            dataset_kwargs.class_name = 'data.CrosscutDataset'
            cd.num_channels, cd.condition_dim = 4, 66
            dataset_kwargs.update(project = config.project, num_channels=cd.num_channels, condition_dim=cd.condition_dim)
        dataloader_kwargs = edict(pin_memory=True, num_workers=4, prefetch_factor=2)

        network_kwargs = edict()
        network_kwargs.class_name = 'network.transformer.TransformerModel'
        network_kwargs.update(mid_channels=cn.mid_channels)
        interface_kwargs = dict(in_channels=cd.num_channels, condition_channels=cd.condition_dim, out_channels=cd.num_channels)
        net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs).to(torch.device("cuda"))

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load network.
    if network_pkl is not None:
        dist.print0(f'Loading network from "{network_pkl}"...')
        with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
            ema = pickle.load(f)['ema'].to(torch.device("cuda"))

    if model_state is not None:
        dist.print0(f'Loading training state from "{model_state}"...')
        data = torch.load(model_state, map_location=torch.device('cpu'))
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        del data # conserve memory

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Select batch size per GPU.
    batch_gpu = batch_size // dist.get_world_size()
    
    # Load dataset.
    dist.print0('Loading dataset...')
    test_dataset = dnnlib.util.construct_class_by_name(split="test", **dataset_kwargs)
    test_dataloader = iter(torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_gpu, **dataloader_kwargs))

    diffusion_kwargs = edict()
    diffusion_kwargs.class_name = 'gaussian_diffusion.GaussianDiffusion'
    diffusion_kwargs.update(num_timesteps=cf.num_timesteps, noise_schedule=cf.noise_schedule, predict_xstart=cf.predict_xstart)
    diffuser = dnnlib.util.construct_class_by_name(**diffusion_kwargs)

    # Description string.
    desc = f'{config.exp_name:s}'

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
    log_kwargs = edict()
    log_kwargs.class_name = 'log.Logger'
    log_kwargs.update(project_name = config.project, exp_name = config.exp_name, tags = f'{cur_run_id:05d}-{desc}')
    logger = dnnlib.util.construct_class_by_name(**log_kwargs)

    cc.net = net
    cc.diffuser = diffuser
    cc.dataloader = test_dataloader
    cc.num_timesteps = cf.num_timesteps
    cc.logger = logger

    evaluation_loop(**cc)

def parse_args():
    parser = argparse.ArgumentParser(description='Equivariant Learning to Shape Puzzles')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)
    args = parser.parse_args()
    update_config(args.cfg)

if __name__ == '__main__':
    main()