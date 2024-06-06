import yaml
from dnnlib import EasyDict as edict

config = edict()
config.project = "ShapeAssembly"
config.exp_name = "puzzlefusion"
config.outdir = "./training-runs"
config.model_pkl = None
config.model_state = None
config.nosubdir = False
config.dry_run = False
config.batch_size = 512
config.total_kobj = 200000   
config.kobj_per_tick = 100         # Interval of progress prints.
config.snapshot_ticks = 1        # How often to save network snapshots, None = disable.
config.state_dump_ticks = 500       # How often to dump training state, None = disable.

config.dataset = edict()

config.network = edict()
config.network.mid_channels = 256

config.diffusion = edict()
config.diffusion.num_timesteps = 1000
config.diffusion.noise_schedule = "cosine"
config.diffusion.predict_xstart = True

config.loss = edict()
config.loss.use_matching_loss = True

config.optimizer = edict()
config.optimizer.lr = 1e-3
config.optimizer.weight_decay = 0.05
config.optimizer.lr_decay_kobj = 100000         # Learning rate decay interval
config.optimizer.lr_rampup_kobj = 10000        # Learning rate ramp-up duration.
config.optimizer.ema_halflife_kobj = 500       # Half-life of the exponential moving average (EMA) of model weights.
config.optimizer.ema_rampup_ratio = 0.05      # EMA ramp-up coefficient, None = no rampup.

def _update_dict(k, v):
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))

def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                     config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))