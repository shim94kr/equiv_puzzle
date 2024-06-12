import os
import numpy as np
import csv
from tqdm import tqdm
from collections import defaultdict
from glob import glob

def rotate_points(points, piece_idx):
    rotation_degree = 0 if piece_idx == 0 else np.random.rand() * 360
    rotation_angle = np.deg2rad(rotation_degree)
    rotation_matrix = np.array([
        [np.cos(rotation_angle), -np.sin(rotation_angle)], # this is selected for return
        [np.sin(rotation_angle), np.cos(rotation_angle)]
    ])
    rotated_points = np.matmul(rotation_matrix, points.T).T

    # use cos and sin as gt rotation
    rot = np.array([np.cos(- rotation_angle), np.sin(- rotation_angle)])
    gt_rot = rot[None, :].repeat(rotated_points.shape[0], axis=0)
    return rotated_points, gt_rot

def translate_points(points):
    points = points / 100. - 0.5 # [[x0,y0],[x1,y1],...,[x15,y15]] and map to 0-1 - > -0.5, 0.5
    points = points * 2 # map to [-1, 1]
    center = np.mean(points, 0, keepdims=True)
    points = points - center

    gt_t = center.repeat(points.shape[0], axis=0)
    return points, gt_t

def preprocess_crosscut_data(
    split = 'train',
):
    base_dir = f'./datasets/cross_cut/{split}_poly_data'
    lines_dir = glob(f'{base_dir}/*')
    num_rels = 100
    for directory in lines_dir:
        puzzles_dir = glob(f'{directory}/*')
        for puzzle_name in tqdm(puzzles_dir):
            # get pieces info for each puzzle
            with open(f'{puzzle_name}/ground_truth_puzzle.csv') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')

                piece_dict = defaultdict(list)
                for row in reader:
                    if row[0] == 'piece':
                        continue
                    piece_dict[float(row[0])].append([float(row[1]),float(row[2])])

                pieces = []
                for i, piece in piece_dict.items():
                    pieces.append(np.array(piece))
            
            # get maching info of a puzzle
            # starting point of each piece
            start_points = [0]
            for i in range(len(pieces)-1):
                start_points.append(start_points[-1]+len(pieces[i]))

            with open(f'{puzzle_name}/ground_truth_rels.csv') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                rels = []
                for row in reader:
                    if row[0] == 'piece1':
                        continue
                    [p1, e1, p2, e2] = [int(x) for x in row]
                    p11, p12 = pieces[p1][e1], pieces[p1][(e1+1)%len(pieces[p1])]
                    p21, p22 = pieces[p2][e2], pieces[p2][(e2+1)%len(pieces[p2])]
                    if np.abs(p11-p21).sum()<np.abs(p11-p22).sum():
                        rels.append([start_points[p1]+e1, start_points[p2]+e2])
                        rels.append([start_points[p1]+(e1+1)%(len(pieces[p1])), start_points[p2]+(e2+1)%(len(pieces[p2]))])
                    else:
                        rels.append([start_points[p1]+e1, start_points[p2]+(e2+1)%(len(pieces[p2]))])
                        rels.append([start_points[p1]+(e1+1)%(len(pieces[p1])), start_points[p2]+e2])

                rels = np.array(rels)

            # make dataset for each puzzle
            t, rot, vertices, corner_idx, piece_idx = [], [], [], [], []
            for i, piece in enumerate(pieces):
                # duplicate index of piece by number of corners
                vertices_ = np.array(piece)
                piece_idx_ = np.array([int(i)]).repeat(vertices_.shape[0], axis=0)
                corner_idx_ = np.array(range(len(piece)))

                # normalize / compute center (gt translation) / translate to origin
                vertices_, t_ = translate_points(vertices_)

                # random rotate vertices_
                vertices_, rot_ = rotate_points(vertices_, i)

                vertices.append(vertices_)
                t.append(t_)
                rot.append(rot_)
                piece_idx.append(piece_idx_)
                corner_idx.append(corner_idx_)
            
            piece_idx, corner_idx = np.concatenate(piece_idx, axis=0), np.concatenate(corner_idx, axis=0)
            vertices, t, rot = np.concatenate(vertices, axis=0),np.concatenate(t, axis=0), np.concatenate(rot, axis=0)

            puzzle = {'piece_idx': piece_idx, 'corner_idx': corner_idx, 'vertices': vertices, 't': t, 'rot': rot, 'rels': rels}

            path_dir = f"./datasets/cross_cut/processed_{split}/"
            path_file = os.path.join(path_dir, f"{puzzle_name.split('/')[4]}_{puzzle_name.split('/')[5]}")
            
            os.makedirs(path_dir, exist_ok=True)
            np.savez_compressed(path_file, **puzzle)

if __name__ == '__main__':
    dataset = preprocess_crosscut_data('train')
    dataset = preprocess_crosscut_data('test')