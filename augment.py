import torch
from utils import get_transforms, matrix_to_euler_angles, theta_to_rot_params

# Custom svd_flip function for PyTorch
def svd_flip(u, v):
    batch_size, ndim, num_corners = v.shape
    max_abs_v_cols = torch.argmax(torch.abs(v), axis=2)
    batch_idx = torch.arange(batch_size)[:, None]
    dim_idx = torch.arange(ndim)[None, :]
    signs = torch.sign(v[batch_idx, dim_idx, max_abs_v_cols])
    u *= signs[:, None, :]
    v *= signs[:, :, None]
    
    determinant = torch.det(u)
    u[:, :, 0] *= determinant[:, None]
    v[:, 0, :] *= determinant[:, None]
    return u, v

# Function to perform PCA with consistent sign convention
def svd_with_sign(X):
    # Center the data
    t = torch.mean(X, dim=1, keepdim=True)
    X_centered = X - t

    # Compute SVD
    U, S, Vt = torch.linalg.svd(X_centered.permute(0, 2, 1), full_matrices=False)

    # Ensure consistent sign of eigenvectors using svd_flip
    U, Vt = svd_flip(U, Vt)
    R = U
    X_canonicalized = S[:, None, :] * Vt.permute(0, 2, 1)
    return S, t, R, X_canonicalized 

def vertices_to_structured_pieces(vertices, piece_idx, corner_idx, padding_mask, device, dvalue = 0):
    max_num_pieces = (piece_idx.max(dim=1)[0].max(dim=0)[0] + 1).to(torch.int64).item()
    max_piece_size = (corner_idx.max(dim=1)[0].max(dim=0)[0] + 1).to(torch.int64).item()
    batch_size, corner_size, ndim = vertices.shape

    # Initialize a tensor to store the structured pieces data & introduce additional dimension to account of padding.
    structured_pieces = torch.ones((batch_size, max_num_pieces * (max_piece_size + 1), ndim), dtype=torch.float32) * dvalue
    structured_pieces = structured_pieces.to(device)

    # Create an index tensor for scatter_add
    index = ((piece_idx * max_piece_size + corner_idx) + (1- padding_mask) * (max_num_pieces * max_piece_size))[..., None].repeat(1, 1, ndim).to(torch.int64)
    structured_pieces.scatter_(dim=1, index=index, src=vertices)

    structured_pieces = structured_pieces[:, :max_num_pieces * max_piece_size].reshape(-1, max_num_pieces, max_piece_size, ndim)
    return structured_pieces

class AugmentPipe:
    def __init__(self, 
                 shape_canonicalization = False,
                 anchor_selection = False,
                 corner_connection = False):
        super().__init__()
        self.shape_canonicalization = shape_canonicalization
        self.anchor_selection = anchor_selection
        self.corner_connection = corner_connection

    def __call__(self, data):
        vertices, piece_idx, corner_idx, padding_mask = data['vertices'], data['piece_idx'], data['corner_idx'], data['padding_mask']
        (batch_size, corner_size, ndim), device = vertices.shape, vertices.device
        max_num_pieces = (piece_idx.max(dim=1)[0].max(dim=0)[0] + 1).to(torch.int64).item()
        max_piece_size = (corner_idx.max(dim=1)[0].max(dim=0)[0] + 1).to(torch.int64).item()

        if self.shape_canonicalization:
            # Batch x # of pieces x # of corners x # of dim
            structured_pieces = vertices_to_structured_pieces(vertices, piece_idx, corner_idx, padding_mask, device)
            
            # Apply shape canonicalization
            _, canonical_translation, canonical_rotation_matrix, canonical_pieces = svd_with_sign(structured_pieces.reshape(-1, max_piece_size, ndim))
            canonical_rotation_matrix = canonical_rotation_matrix.reshape(batch_size*max_num_pieces, 1, ndim*ndim).repeat(1, max_piece_size, 1)
            canonical_translation = canonical_translation.reshape(batch_size*max_num_pieces, 1, ndim).repeat(1, max_piece_size, 1)
            canonical_structured_params = torch.cat([canonical_translation, canonical_rotation_matrix, canonical_pieces], dim=-1)
            canonical_structured_params = canonical_structured_params.reshape(batch_size, max_num_pieces*max_piece_size, -1)

            # Initialize a tensor to store the canonical_vertices 
            canonical_dims = canonical_structured_params.shape[-1]
            canonical_params = torch.zeros(batch_size, corner_size+1, canonical_dims).to(device)
            index = ((piece_idx * max_piece_size + corner_idx) + (1- padding_mask) * (max_num_pieces * max_piece_size))[..., None].repeat(1, 1, canonical_dims).to(torch.int64)
            index_src = torch.arange(corner_size)[None,:,None].repeat(batch_size, 1, canonical_dims).to(device)
            inverse_idx = (torch.ones((batch_size, max_num_pieces * (max_piece_size + 1), canonical_dims), dtype=torch.int64) * corner_size).to(device)
            inverse_idx.scatter_(dim=1, index=index, src=index_src)
            inverse_idx = inverse_idx[:, :max_num_pieces * max_piece_size]

            split_dims = [ndim, ndim*ndim, ndim]
            canonical_params.scatter_add_(dim=1, index=inverse_idx, src = canonical_structured_params)
            canonical_params = canonical_params[:, :corner_size] * padding_mask[:, :, None]
            canonical_translation, canonical_rotation_matrix, canonical_vertices = torch.split(canonical_params, split_dims, dim=-1)
            canonical_rotation_matrix = canonical_rotation_matrix.reshape(batch_size, corner_size, ndim, ndim)

            data['vertices'] = canonical_vertices.detach()
            raw_t, raw_rotation_matrix = get_transforms(torch.cat([data["t"], data["rot"]], dim=-1))
            data["t"] = raw_t + torch.einsum('bpde,bpe->bpd', raw_rotation_matrix, canonical_translation).detach()
            rotation_matrix = torch.einsum('bpde,bpef->bpdf', raw_rotation_matrix, canonical_rotation_matrix)
            data["rot"] = theta_to_rot_params(matrix_to_euler_angles(rotation_matrix, "ZYX")).detach()

        if self.anchor_selection:
            vertices = data['vertices']
            structured_pieces_min = vertices_to_structured_pieces(vertices, piece_idx, corner_idx, padding_mask, device, dvalue=999).min(dim=2)[0]
            structured_pieces_max = vertices_to_structured_pieces(vertices, piece_idx, corner_idx, padding_mask, device, dvalue=-999).max(dim=2)[0]

            # based on the length of axes-aligned bounding box we choose anchor piece.
            length = structured_pieces_max - structured_pieces_min
            max_length = length.max(dim=-1)[0]
            anchor_piece = max_length.max(dim=-1)[1]
            
            data['anchor_piece'] = anchor_piece.detach()
            data['anchor_mask'] = torch.tensor(piece_idx == anchor_piece[:,None])[:, :, None].to(device).detach()
        
        if self.corner_connection:
            vertices, rels_mask = data['vertices'], data['rels_mask']
            indices = torch.topk(rels_mask, k=30 // ndim, dim=-1)[1]
            arange = torch.arange(batch_size)[:, None, None].repeat(1, corner_size, 30 // ndim)
            cc = vertices.unsqueeze(2) - vertices[arange, indices]
            data['connected_corners'] = (cc / (cc.norm(dim=-1, keepdim=True) + 1e-8)).flatten(-2, -1).detach()

        return data