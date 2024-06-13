import torch

def anchor_centering(x_t, data, dir=0):
    if not dir:
        piece_mask, anchor_mask = data["piece_mask"], data['anchor_mask']
        x_t = torch.einsum('bcd,bce->bed', x_t, piece_mask) / piece_mask.sum(dim=-1, keepdim=True)
        t, rotation_matrix = get_transforms(x_t, SE_domain=True)
        anchor_t = (t * anchor_mask).sum(dim=1) / anchor_mask.sum(dim=1)
        anchor_rotation_matrix = (rotation_matrix * anchor_mask[:, :, :, None]).sum(dim=1) / anchor_mask.sum(dim=1)[:, :, None]

        new_t = (t - anchor_t[:,None])
        new_t = torch.einsum('bcd,bpd->bpc', anchor_rotation_matrix.permute(0, 2, 1), new_t)
        new_r = torch.einsum('bcd,bpdf->bpcf', anchor_rotation_matrix.permute(0, 2, 1), rotation_matrix)
        new_r = theta_to_rot_params(matrix_to_euler_angles(rotation_matrix, "ZYX"))
        x_t = torch.cat([new_t, new_r], dim = -1)
    else:
        t, rotation_matrix = get_transforms(x_t, SE_domain=True)
        new_t = t
        new_t = torch.einsum('bcd,bpd->bpc', anchor_rotation_matrix, new_t) + anchor_t[:, None]
        new_r = torch.einsum('bcd,bpdf->bpcf', anchor_rotation_matrix, rotation_matrix)
        new_r = theta_to_rot_params(matrix_to_euler_angles(rotation_matrix, "ZYX"))
        x_t = torch.cat([new_t, new_r], dim = -1)
    return x_t

def theta_to_rot_params(theta):
    if theta.shape[-1] == 3:
        # Compute yaw, pitch, and roll using advanced indexing
        yaw = theta[..., 0]
        pitch = theta[..., 1]
        roll = theta[..., 2]

        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
        cos_pitch, sin_pitch = torch.cos(pitch), torch.sin(pitch)
        cos_roll, sin_roll = torch.cos(roll), torch.sin(roll)

        rot_params = torch.stack([cos_yaw, sin_yaw, cos_pitch, sin_pitch, cos_roll, sin_roll], dim=-1)
        
    elif theta.shape[-1] == 1:
        # Compute yaw, pitch, and roll using advanced indexing
        cos_theta = torch.cos(theta[..., 0])
        sin_theta = torch.sin(theta[..., 0])
        rot_params = torch.stack([cos_theta, sin_theta], dim=-1)
    
    return rot_params

def angle_from_tan(axis, other_axis, data, horizontal, tait_bryan):
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])

def index_from_letter(letter):
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")


def matrix_to_euler_angles(matrix, convention):
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if matrix.shape[-1] == 3:
        if len(convention) != 3:
            raise ValueError("Convention must have 3 letters.")
        if convention[1] in (convention[0], convention[2]):
            raise ValueError(f"Invalid convention {convention}.")
        for letter in convention:
            if letter not in ("X", "Y", "Z"):
                raise ValueError(f"Invalid letter {letter} in convention string.")
        if matrix.size(-1) != 3 or matrix.size(-2) != 3:
            raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
        i0 = index_from_letter(convention[0]) # "Z"
        i2 = index_from_letter(convention[2]) # "X"
        tait_bryan = i0 != i2
        if tait_bryan:
            central_angle = torch.asin(
                matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
            )
        else:
            central_angle = torch.acos(matrix[..., i0, i0])

        o = (
            angle_from_tan(
                convention[0], convention[1], matrix[..., i2], False, tait_bryan
            ),
            central_angle,
            angle_from_tan(
                convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
            ),
        )
        if convention == "XYZ":
            return torch.stack(o, -1)
        elif convention == "ZYX":
            return torch.stack(o[::-1], -1)
    else:
        return torch.atan2(matrix[..., 1, 0], matrix[..., 0, 0]).unsqueeze(-1)

def get_transforms(x, SE_domain = False):
    # x is supposed to be puzzle fusion output!
    is_3d = (x.shape[-1] == 9)
    if is_3d:
        t, rot_theta = torch.split(x, [3, 6], dim=-1)
        cos_theta, sin_theta = torch.split(rot_theta.reshape(rot_theta.shape[:-1] + (3, 2, )), [1, 1], dim=-1)
        cos_theta, sin_theta = cos_theta.squeeze(-1), sin_theta.squeeze(-1)
        theta = torch.atan2(sin_theta, cos_theta)

        if SE_domain:
            # if using cos/sin directly above, it is not guaranteed to apply the rotation only
            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
        
        rotation_matrix = torch.eye(3).repeat(x.shape[:-1] + (1, 1,)).to(rot_theta.device)
        # stacking roll & pitch & yaw
        for i in reversed(range(3)):
            f, s = (i+1) % 3, (i+2) % 3
            rotation_matrix_i = torch.eye(3).repeat(x.shape[:-1] + (1, 1,)).to(rot_theta.device)
            rotation_matrix_i[..., f, f], rotation_matrix_i[..., f, s] = cos_theta[..., i], -sin_theta[..., i]
            rotation_matrix_i[..., s, f], rotation_matrix_i[..., s, s] = sin_theta[..., i], cos_theta[..., i]
            rotation_matrix = torch.matmul(rotation_matrix_i, rotation_matrix)

    else:
        t, cos_theta, sin_theta = torch.split(x, [2, 1, 1], dim=-1)
        theta = torch.atan2(sin_theta, cos_theta)
        
        if SE_domain:
            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)

        # need to rotate reversly since gt is set by right-order
        rotation_matrix = torch.stack([
            torch.cat([cos_theta, -sin_theta], dim=-1),
            torch.cat([sin_theta, cos_theta], dim=-1),
        ], dim=-2)
    
    return t, rotation_matrix
