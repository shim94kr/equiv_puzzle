import torch
import numpy as np
from utils import get_transforms, theta_to_rot_params, matrix_to_euler_angles
from gaussian_diffusion import ModelMeanType

def mean_flat(tensor, padding_mask):
    """
    Take the mean over all non-batch dimensions.
    """
    tensor = tensor * padding_mask.unsqueeze(-1)
    tensor = tensor.mean(dim=list(range(1, len(tensor.shape))))/torch.sum(padding_mask, dim=1)
    return tensor

class PFLoss:
    def __init__(self,
                 diffuser,
                 anchor_centering = False,
                 use_matching_loss = False,
                 is_3d = False, 
    ):
        self.diffuser = diffuser
        self.anchor_centering = anchor_centering
        self.use_matching_loss = use_matching_loss
        self.is_3d = is_3d

    def __call__(self, net, data, augment_pipe=None, device=torch.device('cuda')):
        data = augment_pipe(data) if augment_pipe is not None else data
        x_start = torch.cat([data["t"], data["rot"]], dim=-1)
        noise = torch.randn_like(x_start)
        timestep = torch.randint(low=0, high=self.diffuser.num_timesteps, size=(x_start.shape[0],)).to(device)
        x_t = self.diffuser.q_sample(x_start, timestep, noise=noise)
            
        model_output_dec, _ = net(x_t, timestep, data)

        target = {
            ModelMeanType.PREVIOUS_X: self.diffuser.q_posterior_mean_variance(
                x_start=x_start, x_t=x_t, t=timestep
            )[0],
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.diffuser.model_mean_type]

        loss = mean_flat(((target - model_output_dec) ** 2), data['padding_mask'])

        if self.use_matching_loss:
            if self.diffuser.model_mean_type == ModelMeanType.EPSILON:
                pred_xstart = self.diffuser._predict_xstart_from_eps(x_t=x_t, t=timestep, eps=model_output_dec).float()
            elif self.diffuser.model_mean_type == ModelMeanType.START_X:
                pred_xstart = model_output_dec
            else:
                assert False

            t, _, rotation_matrix = get_transforms(pred_xstart)
            final_polys = (torch.einsum('bcde,bce->bcd', rotation_matrix, data['vertices']) + t)
            poly_set1 = final_polys[torch.arange(final_polys.shape[0])[:, None], data['rels'][:,:,0].long()]
            poly_set2 = final_polys[torch.arange(final_polys.shape[0])[:, None], data['rels'][:,:,1].long()]
            tmp_mask_pairwise = (data['rels'].sum(2)>0).long()
            loss += mean_flat(((poly_set1 - poly_set2) ** 2) * (t.unsqueeze(1).unsqueeze(1)<500), tmp_mask_pairwise)

        return loss