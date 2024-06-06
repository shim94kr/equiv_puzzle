import os
import json
import numpy as np
import csv
from tqdm import tqdm
from collections import defaultdict
from glob import glob

def rotate_points(points, piece_idx):
    rotation_degree = [0., 0., 0.] if piece_idx == 0 else np.random.rand(3) * 360
    rotation_angle = np.deg2rad(rotation_degree)
    rotated_points = points
    rot = []
    # stacking roll & pitch & yaw
    for i, angle in enumerate(rotation_angle):
        f, s = (i+1) % 3, (i+2) % 3
        rotation_matrix = np.eye(3)
        rotation_matrix[f, f], rotation_matrix[f, s] = np.cos(angle), -np.sin(angle)
        rotation_matrix[s, f], rotation_matrix[s, s] = np.sin(angle), np.cos(angle)
        rotated_points = np.matmul(rotation_matrix, rotated_points.T).T

    rot = np.stack([np.cos(rotation_angle), np.sin(rotation_angle)], axis=1).flatten()
    # use cos and sin as gt rotation
    gt_rot = rot[None, :].repeat(rotated_points.shape[0], axis=0)
    return rotated_points, gt_rot

def translate_points(points):
    points = points / 10. - 0.5 # [[x0,y0],[x1,y1],...,[x15,y15]] and map to 0-1 - > -0.5, 0.5
    points = points * 2 # map to [-1, 1]
    center = np.mean(points, 0, keepdims=True)
    points = points - center

    gt_t = center.repeat(points.shape[0], axis=0)
    return points, gt_t

def preprocess_crosscut_data(
    split = 'train',
):
    base_dir = f'./datasets/randomcut_3d/{split}_data'
    pieces_dir = glob(f'{base_dir}/*')
    num_rels = 100
    pieces_dir = [dir for dir in pieces_dir if "7pieces" in dir]

    for directory in pieces_dir:
        puzzles_dir = glob(f'{directory}/*')
        for puzzle_name in tqdm(puzzles_dir):
            # get pieces info for each puzzle
            with open(f'{puzzle_name}') as f:
                pieces = json.load(f)

            # get maching info of a puzzle
            # starting point of each piece
            start_points = [0]
            for i in range(len(pieces)-1):
                start_points.append(start_points[-1]+len(pieces[f"frag{i}"]["vertices"]))
            num_vertex = start_points[-1] + len(pieces[f"frag{len(pieces)-1}"]["vertices"])

            rels = []
            for i in range(len(pieces)):
                for j in range(i+1, len(pieces)):
                    vertices_i = pieces[f"frag{i}"]["vertices"]
                    vertices_j = pieces[f"frag{j}"]["vertices"]
                    faces_i = pieces[f"frag{i}"]["faces"]
                    faces_j = pieces[f"frag{j}"]["faces"]
                    edges_i = [[face[k], face[(k+1) % len(face)]] for face in faces_i for k in range(len(face))]
                    edges_j = [[face[k], face[(k+1) % len(face)]] for face in faces_j for k in range(len(face))]
                    for edge_i in edges_i:
                        for edge_j in edges_j:
                            v11, v12 = np.array(vertices_i[edge_i[0]]), np.array(vertices_i[edge_i[1]])
                            v21, v22 = np.array(vertices_j[edge_j[0]]), np.array(vertices_j[edge_j[1]])
                            error1 = np.abs(v11 - v21).sum() + np.abs(v12 - v22).sum()
                            error2 = np.abs(v11 - v22).sum() + np.abs(v12 - v21).sum()

                            if error1 < 1e-5:
                                rels.append([start_points[i]+edge_i[0], start_points[j]+edge_j[0]])
                                rels.append([start_points[i]+edge_i[1], start_points[j]+edge_j[1]])
                            elif error2 < 1e-5:
                                rels.append([start_points[i]+edge_i[0], start_points[j]+edge_j[1]])
                                rels.append([start_points[i]+edge_i[1], start_points[j]+edge_j[0]])
            rels = np.array(rels)

            # make dataset for each puzzle
            t, rot, vertices, faces_mask, faces_piece_idx, corner_idx, piece_idx = [], [], [], [], [], [], []
            for i in range(len(pieces)):
                # duplicate index of piece by number of corners
                vertices_ = np.array(pieces[f"frag{i}"]["vertices"])
                faces_ = pieces[f"frag{i}"]["faces"]
                piece_idx_ = np.array([int(i)]).repeat(vertices_.shape[0], axis=0)
                corner_idx_ = np.array(range(len(vertices_)))

                # normalize / compute center (gt translation) / translate to origin
                vertices_, t_ = translate_points(vertices_)

                # random rotate vertices_
                vertices_, rot_ = rotate_points(vertices_, i)

                vertices.append(vertices_)
                t.append(t_)
                rot.append(rot_)
                piece_idx.append(piece_idx_)
                corner_idx.append(corner_idx_)

                faces_mask_ = []
                for face_ in faces_:
                    mask = np.zeros((num_vertex))
                    mask[start_points[i] + np.array(face_)] = 1
                    faces_mask_.append(mask)
                faces_mask.append(np.stack(faces_mask_, axis=0))

                faces_piece_idx_ = np.array([int(i)]).repeat(len(faces_mask_), axis=0)
                faces_piece_idx.append(faces_piece_idx_)
            
            piece_idx, corner_idx = np.concatenate(piece_idx, axis=0), np.concatenate(corner_idx, axis=0)
            faces_piece_idx = np.concatenate(faces_piece_idx, axis=0)
            vertices, faces_mask = np.concatenate(vertices, axis=0), np.concatenate(faces_mask, axis=0)
            t, rot = np.concatenate(t, axis=0), np.concatenate(rot, axis=0)
            
            puzzle = {'piece_idx': piece_idx, 'corner_idx': corner_idx, 'vertices': vertices, 
                    'faces_mask': faces_mask, 'faces_piece_idx': faces_piece_idx,
                    't': t, 'rot': rot, 'rels': rels}
            path_dir = f"./datasets/randomcut_3d/processed_{split}/"
            path_file = os.path.join(path_dir, f"{puzzle_name.split('/')[4]}_{puzzle_name.split('/')[5]}").replace("json", "npz")
            
            os.makedirs(path_dir, exist_ok=True)
            np.savez_compressed(path_file, **puzzle)

if __name__ == '__main__':
    dataset = preprocess_crosscut_data('train')
    dataset = preprocess_crosscut_data('test')