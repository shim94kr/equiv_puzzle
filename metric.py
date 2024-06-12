import torch
import trimesh
import numpy as np

import cv2
from shapely.geometry import LineString

def transform_pieces(polys, batch_samples, piece_masks, is_3d=False):
    # aggregate prediction per piece
    # samples: [# of batches, # of samples, # of corners, t/Rot]
    # piece_masks: [# of batches, # of corners, # of corners]
    batch_samples = torch.einsum('bscd,bce->bsed', batch_samples, piece_masks) / piece_masks.sum(dim=-1, keepdim=True)[:, None]
    if is_3d:
        t, rot_theta = torch.split(batch_samples, [3, 6], dim=-1)
        cos_theta, sin_theta = torch.split(rot_theta.reshape(rot_theta.shape[:-1] + (3, 2, )), [1, 1], dim=-1)
        cos_theta, sin_theta = cos_theta.squeeze(-1), sin_theta.squeeze(-1)
        # if using cos/sin directly above, it is not guaranteed to apply the rotation only
        theta = torch.atan2(sin_theta, cos_theta)
        cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
        
        rotation_matrix = torch.eye(3).repeat(batch_samples.shape[:-1] + (1, 1,)).to(rot_theta.device)
        # stacking roll & pitch & yaw
        for i in reversed(range(3)):
            f, s = (i+1) % 3, (i+2) % 3
            rotation_matrix_i = torch.eye(3).repeat(batch_samples.shape[:-1] + (1, 1,)).to(rot_theta.device)
            rotation_matrix_i[..., f, f], rotation_matrix_i[..., f, s] = cos_theta[..., i], -sin_theta[..., i]
            rotation_matrix_i[..., s, f], rotation_matrix_i[..., s, s] = sin_theta[..., i], cos_theta[..., i]
            rotation_matrix = torch.matmul(rotation_matrix_i, rotation_matrix)

    else:
        t, cos_theta, sin_theta = torch.split(batch_samples, [2, 1, 1], dim=-1)

        theta = torch.atan2(sin_theta, cos_theta)
        cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)

        # need to rotate reversly since gt is set by right-order
        rotation_matrix = torch.stack([
            torch.cat([cos_theta, -sin_theta], dim=-1),
            torch.cat([sin_theta, cos_theta], dim=-1),
        ], dim=-2)

    # [# of batches, # of samples, # of corners, 3, 3]
    batch_samples = (torch.einsum('bscde,bce->bscd', rotation_matrix, polys) + t)
    return batch_samples

def get_pieces(batch_samples, piece_idx, padding_mask):
    batch_samples_pieces = []
    for b, samples in enumerate(batch_samples): 
        samples_pieces = []
        for s, sample in enumerate(samples):
            unique_piece = torch.unique(piece_idx).int()
            pieces = []
            for idx in unique_piece:
                # recover coordinate of polygon to 0 ~ 20
                polygon = sample[((piece_idx[b] == idx) * padding_mask[b]).bool()]
                pieces.append(polygon.detach().cpu().numpy())
            samples_pieces.append(pieces)
        batch_samples_pieces.append(samples_pieces)
    return batch_samples_pieces

def get_pieces_3d(batch_samples, faces_mask, faces_piece_idx, faces_padding_mask, piece_idx, padding_mask):
    batch_samples_meshes = []
    for b, samples in enumerate(batch_samples):
        samples_meshes = []
        for s, sample in enumerate(samples):
            unique_piece = torch.unique(piece_idx[b]).int()
            pieces_meshes = []
            for idx in unique_piece:
                # Vertex index mapping
                vertices = sample[((piece_idx[b] == idx) * padding_mask[b]).bool()]
                vertices = vertices.detach().cpu().numpy()
                nonzero_idx = torch.nonzero((piece_idx[b] == idx) * padding_mask[b]).flatten()
                idx_mapping = {idx.item(): pos for pos, idx in enumerate(nonzero_idx)}

                # Create faces vector
                faces_mask_ = faces_mask[b][((faces_piece_idx[b] == idx) * faces_padding_mask[b]).bool()]
                faces = [torch.nonzero(row).squeeze(dim=1).detach().cpu().tolist() for row in faces_mask_]
                faces_triangle = np.array([[idx_mapping[face[0]], idx_mapping[face[i]], idx_mapping[face[i+1]]] 
                                  for face in faces for i in range(1, len(face)-1)])
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces_triangle)
                mesh.fix_normals()
                pieces_meshes.append(mesh)
                
            samples_meshes.append(pieces_meshes)
        batch_samples_meshes.append(samples_meshes)
    return batch_samples_meshes

def compute_metric(gt_pieces, pred_pieces, is_3d=False):
    overlap_score, connection_precision, connection_recall = [], [], []
    for b, (s1, s2) in enumerate(zip(gt_pieces, pred_pieces)):
        R, t = translate_to_gt(s1, s2, is_3d)
        if is_3d:
            s1_vertices = [np.array(p.vertices) for p in s1]
            sol_transformed = [np.array(p.vertices) @ R + t for p in s2]
            for idx in range(len(s2)):
                s2[idx].vertices = sol_transformed[idx]
        else:
            s1_vertices = s1
            sol_transformed = [p @ R + t for p in s2]
            s2 = sol_transformed
        connection_score_index = connection_score(s1_vertices, sol_transformed)
        overlap_score.append(overlap_score_fn(s1, s2, piece_weights(s1, is_3d), is_3d))
        connection_precision.append(connection_score_index[0])
        connection_recall.append(connection_score_index[1])
    overlap_score, connection_precision, connection_recall = np.stack(overlap_score), np.stack(connection_precision), np.stack(connection_recall)
    metric = {
        'overlap': overlap_score,
        'connection_precision': connection_precision,
        'connection_recall': connection_recall,
    }
    return metric

def weighted_points(pieces, is_3d=False): #그냥 여기서는 같은 weight로 줌 = center

    if is_3d == False:
        measures = [np.full((len(piece), 1), cv2.contourArea(piece)) for piece in pieces]
        points = np.vstack(pieces)
    else:
        measures, points = [], []
        for piece in pieces:
            measures.append(np.full((len(piece.vertices), 1), np.abs(piece.volume.astype(np.float32))))
            points.append(np.array(piece.vertices))
        points = np.vstack(points)

    weights = np.vstack(measures)
    return (weights, points)

def translate_to_gt(pieces_gt, pieces_sol, is_3d = False):
    #piece 1개당
    if is_3d == False:
        W, p_gt = weighted_points(pieces_gt)
        p_sol = np.vstack(pieces_sol)
    else:
        W, p_gt = weighted_points(pieces_gt, is_3d)
        p_sol = np.vstack([np.array(piece.vertices) for piece in pieces_sol])
    center_gt, center_sol = [np.sum(W * p, axis=0) / np.sum(W)
                            for p in [p_gt, p_sol]]
    X = p_sol - center_sol
    Y = p_gt - center_gt
    S = X.T @ np.diag(W.squeeze()) @ Y
    U, _, V = np.linalg.svd(S)
    if is_3d == False:
        R = (V @ np.array([[1, 0],
                        [0, np.linalg.det(V @ U.T)]]) @ U.T)
    else:  
        R = (V @ np.array([[1, 0, 0],
                [0, 1, 0],
                [0, 0, np.linalg.det(V @ U.T)]]) @ U.T)
    t = center_gt - center_sol @ R.T
    return (R.T, t)

def piece_weights(pieces, is_3d = False):
    if is_3d == False:
        measures = [cv2.contourArea(p.astype(np.float32)) for p in pieces]
    else:
        measures = [np.abs(p.volume.astype(np.float32)) for p in pieces]
    total = sum(measures)
    return [x / total for x in measures]

def overlap_score_fn(pieces_gt, sol_transformed, weights, is_3d = False):
    total_score = 0
    for p_gt, p_sol, w in zip(pieces_gt, sol_transformed, weights):
        if is_3d == False:
            overlap, _ = cv2.intersectConvexConvex(p_gt.astype(np.float32),
                                        p_sol.astype(np.float32))
            
            curr_measures = cv2.contourArea(p_sol.astype(np.float32))
        else:
            try:
                overlap = p_gt.intersection(p_sol).volume.astype(np.float32)
            except:
                # each mesh is proven to be watertight and normal
                # but the intersection become none and fail to form mesh.
                overlap = 0.
            curr_measures = np.abs(p_sol.volume.astype(np.float32))

        if(curr_measures > 0):
            score = overlap / curr_measures
            total_score += score * w
    return total_score


def connection_score(gt, pred):
    assert len(gt)==len(pred)
    connections_gt = []
    connections_pred = []
    for i in range(len(gt)):
        for j in range(i+1, len(gt)):
            for k in range(gt[i].shape[0]):
                for l in range(gt[j].shape[0]):
                    p1 = LineString([gt[i][k], gt[i][(k+1)%gt[i].shape[0]]])
                    p2 = LineString([gt[j][l], gt[j][(l+1)%gt[j].shape[0]]])
                    if p1.hausdorff_distance(p2) < 0.1:
                        connections_gt.append([i, k, j, l])
                    p1 = LineString([pred[i][k], pred[i][(k+1)%pred[i].shape[0]]])
                    p2 = LineString([pred[j][l], pred[j][(l+1)%pred[j].shape[0]]])
                    if p1.hausdorff_distance(p2) < 0.1:
                        connections_pred.append([i, k, j, l])
    precision = np.sum([connections_pred[x] in connections_gt for x in range(len(connections_pred))])/max(len(connections_pred), 1)
    recall = np.sum([connections_pred[x] in connections_gt for x in range(len(connections_pred))])/max(len(connections_gt), 1)
    return np.array([precision, recall])