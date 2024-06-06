import torch
import numpy as np
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
                 use_matching_loss = False,
                 is_3d = False, 
    ):
        self.diffuser = diffuser
        self.use_matching_loss = use_matching_loss
        self.is_3d = is_3d
        # anchor centering
        # anchor index
        # corner connectivity
        
    def __call__(self, net, data, augment_pipe=None, device=torch.device('cuda')):
        # data = augment_pipe(data)
        # anchor centering
        x_start = torch.cat([data["t"], data["rot"]], dim=-1)

        noise = torch.randn_like(x_start)
        t = torch.randint(low=0, high=self.diffuser.num_timesteps, size=(x_start.shape[0],)).to(device)
        x_t = self.diffuser.q_sample(x_start, t, noise=noise)
        
        model_output_dec, _ = net(x_t, t, data)

        target = {
            ModelMeanType.PREVIOUS_X: self.diffuser.q_posterior_mean_variance(
                x_start=x_start, x_t=x_t, t=t
            )[0],
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.diffuser.model_mean_type]

        loss = mean_flat(((target - model_output_dec) ** 2), data['padding_mask'])

        if self.use_matching_loss:
            if self.diffuser.model_mean_type == ModelMeanType.EPSILON:
                pred_xstart = self.diffuser._predict_xstart_from_eps(x_t=x_t, t=t, eps=model_output_dec).float()
            elif self.diffuser.model_mean_type == ModelMeanType.START_X:
                pred_xstart = model_output_dec
            else:
                assert False

            if self.is_3d:
                t, rot_theta = torch.split(pred_xstart, [3, 6], dim=-1)
                cos_theta, sin_theta = torch.split(rot_theta.reshape(rot_theta.shape[:-1] + (3, 2, )), [1, 1], dim=-1)
                cos_theta, sin_theta = cos_theta.squeeze(-1), sin_theta.squeeze(-1)
                
                rotation_matrix = torch.eye(3).repeat(pred_xstart.shape[:-1] + (1, 1,)).to(rot_theta.device)
                # stacking roll & pitch & yaw
                for i in reversed(range(3)):
                    f, s = (i+1) % 3, (i+2) % 3
                    rotation_matrix_i = torch.eye(3).repeat(pred_xstart.shape[:-1] + (1, 1,)).to(rot_theta.device)
                    rotation_matrix_i[..., f, f], rotation_matrix_i[..., f, s] = cos_theta[..., i], sin_theta[..., i]
                    rotation_matrix_i[..., s, f], rotation_matrix_i[..., s, s] = -sin_theta[..., i], cos_theta[..., i]
                    rotation_matrix = torch.matmul(rotation_matrix_i, rotation_matrix)

            else:
                t, cos_theta, sin_theta = torch.split(pred_xstart, [2, 1, 1], dim=-1)

                # need to rotate reversly since gt is set by right-order
                rotation_matrix = torch.stack([
                    torch.cat([cos_theta, sin_theta], dim=-1),
                    torch.cat([-sin_theta, cos_theta], dim=-1),
                ], dim=-2)

            final_polys = (torch.einsum('bcde,bce->bcd', rotation_matrix, data['vertices']) + t)
            poly_set1 = final_polys[torch.arange(final_polys.shape[0])[:, None], data['rels'][:,:,0].long()]
            poly_set2 = final_polys[torch.arange(final_polys.shape[0])[:, None], data['rels'][:,:,1].long()]
            tmp_mask_pairwise = (data['rels'].sum(2)>0).long()
            loss += mean_flat(((poly_set1 - poly_set2) ** 2) * (t.unsqueeze(1).unsqueeze(1)<500), tmp_mask_pairwise)

        return loss