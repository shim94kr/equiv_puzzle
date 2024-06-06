import os
import io
import numpy as np
import torch
import imageio
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from metric import transform_pieces, get_pieces_3d
from data import CrosscutDataset3D
from dnnlib import EasyDict as edict

ID_COLOR = {1: "#A19E74",2: "#B18463", 3:"#E8D7FF",4: "#FEFBA3",
                5: "#292A39",6: "#FFD2FC",7: "#68D1CC",8: "#FC696B",
                9: "#D9BA8B",10: "#232B33",11: "#D3E7D0",12: "#39272F",
                13: "#33443E",14: "#7B813D",15: "#E980FC",16: "#D65E2E",
                17:"#D57C59",18: "#8E838C",19: "#3F3052" ,20:"#043E5F",
                21: "#8CD0A1",22: "#C1DBAE",23: "#B96AC9",24: "#231B1B",
                25: "#640D0E" ,26: "#D3B675" ,27:"#82A07E" ,28:"#B89C6F" }

def draw_pieces(run_dir, cur_nobj, name, batch_samples):
    # Create output directory
    out_dir = os.path.join(run_dir, f"outputs-{cur_nobj//1000:06d}")
    os.makedirs(out_dir, exist_ok=True)

    for b, samples in enumerate(batch_samples): 
        with imageio.get_writer(os.path.join(out_dir, f"{name}_sample{b:03d}.gif"), mode='I', duration=0.5) as writer:
            for s, sample in enumerate(samples):
                fig, ax = plt.subplots()
                for idx, polygon in enumerate(sample):
                    # recover coordinate of polygon to 0 ~ 20
                    polygon = (polygon + 1.0) * 10
                    polygon = polygon.detach().cpu().numpy()
                    polygon = Polygon(polygon, closed=True, fill=True, edgecolor='k', facecolor=ID_COLOR[idx.item() + 1])
                    ax.add_patch(polygon)

                ax.set_xlim(0, 25)
                ax.set_ylim(0, 25)
                buf = io.BytesIO()
                plt.show()
                plt.savefig(buf, format='png')
                plt.close(fig)
                buf.seek(0)
                image = imageio.v2.imread(buf)
                writer.append_data(image)

def draw_pieces_3d(run_dir, cur_nobj, name, batch_samples_mesh):
    # Create output directory
    out_dir = os.path.join(run_dir, f"outputs-{cur_nobj//1000:06d}")
    os.makedirs(out_dir, exist_ok=True)

    for b, samples_meshes in enumerate(batch_samples_mesh): 
        with imageio.get_writer(os.path.join(out_dir, f"{name}_sample{b:03d}.gif"), mode='I', duration=0.5) as writer:
            #for sample_faces, sample_vertices in zip(samples_faces, samples_vertices):
            for sample_meshes in samples_meshes:
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, projection='3d')
                all_pieces_vertices = []
                for i, piece_meshes in enumerate(sample_meshes):
                    color = ID_COLOR[i+1]
                    piece_vertices, piece_faces = piece_meshes.vertices, piece_meshes.faces
                    faces_vertices = piece_vertices[piece_faces]
                    all_pieces_vertices.append(piece_vertices)
                    '''
                    # to visualize without triangle, use this code!
                    # but, current generation process always output triangular faces.
                    faces_vertices_wo_triangle, face_vertices = [], [piece_vertices[piece_faces[0]]]
                    for j, idx_vertices in enumerate(piece_faces[1:]):
                        if not (idx_vertices[0] == piece_faces[j][0] and idx_vertices[1] == piece_faces[j][2]):
                            faces_vertices_wo_triangle.append(np.concatenate(face_vertices, axis=0))
                            face_vertices = [piece_vertices[idx_vertices]]
                        face_vertices.extend([piece_vertices[idx_vertices][-1][None,:]])
                    faces_vertices_wo_triangle.append(np.concatenate(face_vertices, axis=0))
                    '''
                    # Get list of vertices of each face
                    collection = Poly3DCollection(faces_vertices, alpha=0.7, edgecolor='k', facecolor=color)
                    ax.add_collection3d(collection)
                    ax.scatter(piece_vertices[:, 0], piece_vertices[:, 1], piece_vertices[:, 2], c='b', marker='o')
                
                all_pieces_vertices = np.concatenate(all_pieces_vertices, axis=0)
                coords_min, coords_max= all_pieces_vertices.min(axis=0), all_pieces_vertices.max(axis=0)
                limits = np.stack([coords_min, coords_max], axis=1)
                # Render the scene to an image
                ax.set_xlim(limits[0])
                ax.set_ylim(limits[1])
                ax.set_zlim(limits[2])

                buf = io.BytesIO()
                plt.show()
                plt.savefig(buf, format='png')
                plt.close(fig)
                buf.seek(0)
                image = imageio.v2.imread(buf)
                writer.append_data((np.asarray(image) * 255.).astype(np.uint8))

                '''
                # Please use below code instead upper one to visualize 360-degree view
                num_frames=36
                for angle in np.linspace(0, 360, num_frames):
                    ax.view_init(30, angle)
                    plt.tight_layout()
                    buf = io.BytesIO()
                    plt.show()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    image = imageio.v2.imread(buf)
                    writer.append_data((np.asarray(image) * 255.).astype(np.uint8))
                '''

def draw_pieces_3d_old(run_dir, cur_nobj, name, batch_samples, faces_mask, faces_piece_idx, faces_padding_mask, piece_idx, padding_mask):
    # Create output directory
    out_dir = os.path.join(run_dir, f"outputs-{cur_nobj//1000:06d}")
    os.makedirs(out_dir, exist_ok=True)

    for b, samples in enumerate(batch_samples): 
        with imageio.get_writer(os.path.join(out_dir, f"{name}_sample{b:03d}.gif"), mode='I', duration=0.5) as writer:
            for s, sample in enumerate(samples):
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, projection='3d')
                coords_min, coords_max= sample[padding_mask[b].bool()].min(axis=0)[0], sample[padding_mask[b].bool()].max(axis=0)[0]
                limits = torch.stack([coords_min, coords_max], dim=1).detach().cpu().numpy()
                unique_piece = torch.unique(piece_idx).int()
                for idx in unique_piece:
                    # Vertex index mapping
                    vertices = sample[((piece_idx[b] == idx) * padding_mask[b]).bool()]
                    vertices = vertices.detach().cpu().numpy()
                    nonzero_idx = torch.nonzero((piece_idx[b] == idx) * padding_mask[b]).flatten()
                    idx_mapping = {idx.item(): pos for pos, idx in enumerate(nonzero_idx)}
                    # Create faces vector
                    faces_mask_ = faces_mask[b][((faces_piece_idx[b] == idx) * faces_padding_mask[b]).bool()]
                    faces = [torch.nonzero(row).squeeze(dim=1).detach().cpu().tolist() for row in faces_mask_]
                    color = ID_COLOR[idx.item() + 1]
                    faces_vertices = []
                    for face in faces:
                        face_vertices = [vertices[idx_mapping[idx]] for idx in face]
                        face_vertices = np.stack(face_vertices, 0)
                        faces_vertices.append(face_vertices)
                    # Get list of vertices of each face
                    collection = Poly3DCollection(faces_vertices, alpha=0.7, edgecolor='k', facecolor=color)
                    ax.add_collection3d(collection)

                    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='b', marker='o')
                
                # Render the scene to an image
                ax.set_xlim(limits[0])
                ax.set_ylim(limits[1])
                ax.set_zlim(limits[2])

                buf = io.BytesIO()
                plt.show()
                plt.savefig(buf, format='png')
                plt.close(fig)
                buf.seek(0)
                image = imageio.v2.imread(buf)
                writer.append_data((np.asarray(image) * 255.).astype(np.uint8))

                '''
                # Please use below code instead upper one to visualize 360-degree view
                num_frames=36
                for angle in np.linspace(0, 360, num_frames):
                    ax.view_init(30, angle)
                    plt.tight_layout()
                    buf = io.BytesIO()
                    plt.show()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    image = imageio.v2.imread(buf)
                    writer.append_data((np.asarray(image) * 255.).astype(np.uint8))
                '''

# Define the function to create a triangle mesh with unique trajectories
def create_mesh(vertices, faces, piece_idx):
    # Create a triangle mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    faces_triangle = [[face[0], face[i], face[i+1]] for face in faces for i in range(1, len(face)-1)]
    mesh.triangles = o3d.utility.Vector3iVector(faces_triangle)
    
    # Set vertex colors
    color = ID_COLOR[piece_idx.item() + 1]
    color = tuple(int(color[i:i+2], 16) / 255. for i in (1, 3, 5)) 
    mesh.vertex_colors = o3d.utility.Vector3dVector([color] * len(vertices))
    import pdb; pdb.set_trace()
    
    return mesh

def draw_pieces_3d_o3d(run_dir, cur_nobj, name, batch_samples, faces_mask, faces_piece_idx, faces_padding_mask, piece_idx, padding_mask):
    # Create output directory
    out_dir = os.path.join(run_dir, f"outputs-{cur_nobj//1000:06d}")
    os.makedirs(out_dir, exist_ok=True)

    # Create an offscreen renderer
    width, height = 800, 600
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

    # Set the background color to white (RGB: 255, 255, 255)
    background_color = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)
    renderer.scene.set_background(background_color)

    # Set up camera parameters (fixed viewpoint)
    camera = renderer.scene.camera
    bounds = o3d.geometry.AxisAlignedBoundingBox([0, 0, 0], [5, 5, 5])
    center = bounds.get_center()
    extent = bounds.get_extent()
    # Defines where to look at, where to be positioned, where to be directed in addition to viewpoint.
    camera.look_at(center, center + [0, 0, extent[2]], [0, 1, 0])

    for b, samples in enumerate(batch_samples): 
        with imageio.get_writer(os.path.join(out_dir, f"{name}_sample{b:03d}.gif"), mode='I', duration=0.5) as writer:
            renderer.scene.clear_geometry()
            for s, sample in enumerate(samples):
                unique_piece = torch.unique(piece_idx).int()
                for idx in unique_piece:
                    # Vertex index mapping
                    vertices = sample[((piece_idx[b] == idx) * padding_mask[b]).bool()]
                    vertices = vertices.detach().cpu().numpy()
                    nonzero_idx = torch.nonzero((piece_idx[b] == idx) * padding_mask[b]).flatten()
                    idx_mapping = {idx.item(): pos for pos, idx in enumerate(nonzero_idx)}
                    # Create faces vector
                    faces_mask_ = faces_mask[b][((faces_piece_idx[b] == idx) * faces_padding_mask[b]).bool()]
                    faces = [torch.nonzero(row).squeeze(dim=1).detach().cpu().tolist() for row in faces_mask_]
                    faces = [[idx_mapping[idx] for idx in face] for face in faces ]
                    mesh = create_mesh(vertices, faces, idx)
                    
                    # Assign defaultLit shader to the mesh for shading
                    mesh.compute_vertex_normals()
                    
                    material = o3d.visualization.rendering.MaterialRecord()
                    material.shader = "defaultLit"
                    renderer.scene.add_geometry(f"mesh_{idx}", mesh, material)
            # Render the scene to an image
            image = renderer.render_to_image()
            writer.append_data((np.asarray(image) * 255.).astype(np.uint8))

if __name__ == '__main__':
    dataset = CrosscutDataset3D('puzzlefusion', 'train')
    dataloader_kwargs = edict(pin_memory=True, num_workers=4, prefetch_factor=2)
    dataloader = iter(torch.utils.data.DataLoader(dataset=dataset, batch_size=4, **dataloader_kwargs))
    data = next(dataloader)
    
    x_gt = torch.cat([data["t"], data["rot"]], dim=-1)
    sample_gt = x_gt.unsqueeze(1)
    gt_piece = transform_pieces(data['vertices'], sample_gt, data["piece_mask"], is_3d=True)
    #gt_faces, gt_vertices = get_pieces_3d(gt_piece, data["faces_mask"], data["faces_piece_idx"], data["faces_padding_mask"], data["piece_idx"], data["padding_mask"])
    gt_meshes = get_pieces_3d(gt_piece, data["faces_mask"], data["faces_piece_idx"], data["faces_padding_mask"], data["piece_idx"], data["padding_mask"])
    draw_pieces_3d('./', 0, 'gt', gt_meshes)
    import pdb; pdb.set_trace()