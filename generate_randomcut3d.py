import os
import json
import trimesh
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from scipy.spatial import ConvexHull

# 랜덤한 3D 포인트 생성
def generate_random_points(num_points=30, scale=10):
    return np.random.rand(num_points, 3) * scale

# Convex Hull 생성
def create_convex_hull(points):
    hull = ConvexHull(points)
    return hull

# 랜덤 평면 생성
def generate_random_plane(scale=10):
    point = np.random.rand(3) * scale
    normal = np.random.randn(3)
    normal /= np.linalg.norm(normal)
    return point, normal

# 평면으로 자르기
def slice_mesh_with_plane(mesh, point, normal):
    plane_origin = np.array(point)
    plane_normal = np.array(normal)
    
    # 평면으로 메쉬 자르기
    sliced_mesh = trimesh.intersections.slice_mesh_plane(mesh, plane_normal, plane_origin, cap=True)
    final_mesh=fix_mesh(sliced_mesh)
    
    return final_mesh

# 메쉬 고치기
def fix_mesh(mesh):
    # 면 방향 수정
    mesh.fix_normals()
    # 구멍 채우기
    mesh.fill_holes()

    return mesh

# 여러 평면으로 자르기
def slice_mesh(mesh, piece_num=8):
    meshes=[mesh]
    threshold=0.4
    while 1:
        new_meshes=[]
        for mesh in meshes:
            if piece_num-len(new_meshes)==1:
                new_meshes.append(mesh)
                return new_meshes
            
            #mesh 잘 자를때까지 못나감
            while True:
                point, normal = generate_random_plane()
                mesh1=slice_mesh_with_plane(mesh, point, normal)
                mesh2=slice_mesh_with_plane(mesh, point, normal*-1)
                if mesh1.is_empty==False and mesh2.is_empty==False:
                    if min(mesh1.volume, mesh2.volume)/(mesh1.volume+mesh2.volume) < threshold:
                        continue
                    new_meshes.append(mesh1)
                    new_meshes.append(mesh2)
                    break

            if len(new_meshes)==piece_num:
                return new_meshes
            
        meshes=new_meshes

def is_same_plane(plane1, plane2, tol=1e-3):
    # 두 평면의 법선 벡터가 거의 평행한지 확인
    normal1, d1 = plane1[:3], plane1[3]
    normal2, d2 = plane2[:3], plane2[3]
    
    if np.allclose(normal1, normal2, atol=tol) or np.allclose(normal1, -normal2, atol=tol):
        # 한 평면의 점이 다른 평면의 방정식을 만족하는지 확인
        return np.abs(d1 - d2) < tol
    return False

def group_triangles_by_plane(mesh, tol=1e-3):
    # 각 삼각형의 평면 방정식 계산
    normals = mesh.face_normals
    ds = (mesh.triangles[:,:,:] * mesh.face_normals[:,None,:]).sum(axis=1).mean(axis=1, keepdims=True)
    planes = np.concatenate([normals, ds], axis=1)

    groups = []
    grouped = np.zeros(len(planes), dtype=bool)
    
    for i, plane in enumerate(planes):
        if np.linalg.norm(normals[i]) < 1e-15:
            continue

        if grouped[i]:
            continue

        group = [i]
        grouped[i] = True
        
        for j in range(i + 1, len(planes)):
            if grouped[j]:
                continue
            
            if is_same_plane(plane, planes[j], tol):
                group.append(j)
                grouped[j] = True
        
        groups.append(group)
    return groups

def find_unique_edges(faces):
    edge_count = defaultdict(int)
    for face in faces:
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i+1)%3]]))
            edge_count[edge] += 1
    unique_edges = [edge for edge, count in edge_count.items() if count == 1]
    return unique_edges

# 그래프를 정의하는 클래스
class Graph:
    def __init__(self, edges):
        self.graph = defaultdict(list)
        for u, v in edges:
            self.graph[u].append(v)
            self.graph[v].append(u)

    def find_largest_cycle(self):
        visited = set()
        longest_cycle = []

        def dfs(node, parent, path):
            nonlocal longest_cycle
            visited.add(node)
            path.append(node)

            for neighbor in self.graph[node]:
                if neighbor == parent:
                    continue
                if neighbor in visited:
                    # 사이클 감지
                    cycle_start_index = path.index(neighbor)
                    cycle = path[cycle_start_index:]
                    if len(cycle) > len(longest_cycle):
                        longest_cycle = cycle
                else:
                    dfs(neighbor, node, path)

            path.pop()
            visited.remove(node)

        for node in self.graph:
            if node not in visited:
                dfs(node, -1, [])

        return longest_cycle

def point_to_line_distance(point, line_start, line_end):
    # 점과 직선 사이의 거리를 계산하는 함수
    line_vec = line_end - line_start
    point_vec = point - line_start
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / (line_len + 1e-12)
    point_vec_scaled = point_vec / (line_len + 1e-12)
    t = np.dot(line_unitvec, point_vec_scaled)
    nearest = line_start + t * line_vec
    distance = np.linalg.norm(nearest - point)
    return distance

def remove_near_collinear_points(boundary_loop, coords, threshold=1e-3):
    # 주어진 boundary loop에서 직선 가까이 존재하는 점을 제거하는 함수
    filtered_loop = []  # 첫 번째 점을 추가
    
    for i in range(len(boundary_loop)):
        prev_point = coords[boundary_loop[i - 1]]
        curr_point = coords[boundary_loop[i]]
        next_point = coords[boundary_loop[i + 1-len(boundary_loop)]]
        
        distance = point_to_line_distance(curr_point, prev_point, next_point)
        
        if distance >= threshold:
            filtered_loop.append(boundary_loop[i])
    
      # 마지막 점을 추가
    return filtered_loop

def reduce_mesh(mesh, groups):
    new_vertices = []
    new_faces = []
    new_faces_label = []
    new_puzzlefaces=[]
    vertex_map = {}
    current_index = 0
    fail = False
    for group in groups:
        group_faces = mesh.faces[group]
        unique_edges = find_unique_edges(group_faces)
        
        if not unique_edges:
            continue

        # unique edge를 얻었으므로, 이제 이를 기반으로 필요없는 점들만 제거하면 된다
        unique_points = np.unique(np.array(unique_edges).flatten()) #index
        graph = Graph(unique_edges)
        boundary_loop = graph.find_largest_cycle() #index 
        if len(boundary_loop)!=len(unique_points):
            print("Not consistent")
            raise ValueError
        
        # 점들을 돌아가면서 직선상 놓인것 제거
        boundary_loop= remove_near_collinear_points(boundary_loop, mesh.vertices)
        
        # 남은 정점에 대한 인덱스 부여, 면들은 인덱스로 관리됨.
        for vertex_idx in boundary_loop:
            if vertex_idx not in vertex_map:
                vertex_map[vertex_idx] = current_index
                new_vertices.append(mesh.vertices[vertex_idx])
                current_index += 1
        
        # 정점 인덱스로 면 정보 저장, 면은 3각형으로 저장
        for i in range(1, len(boundary_loop)-1):
            new_faces.append([vertex_map[boundary_loop[0]],
                              vertex_map[boundary_loop[i]],
                              vertex_map[boundary_loop[i+1]]])

        new_puzzlefaces.append([vertex_map[bp] for bp in boundary_loop])

    return new_vertices, new_puzzlefaces

def validate_mesh(mesh):
    if mesh.is_empty:
        print("Mesh is empty.")
        return False
    if not mesh.is_watertight:
        print("Mesh is not watertight.")
        return False
    if not mesh.is_winding_consistent:
        print("Mesh winding is not consistent.")
        return False
    if not mesh.is_volume:
        print("Mesh does not represent a valid volume.")
        return False
    return True

def reduce_meshes(sliced_meshes, check_mesh=False):
    pieces={}
    for k, sliced_mesh in enumerate(sliced_meshes):
        groups = group_triangles_by_plane(sliced_mesh)
        try:
            reduced_vertices, reduced_faces = reduce_mesh(sliced_mesh, groups)
            # 정점 인덱스로 면 정보 저장, 면은 3각형으로 저장
            reduced_triangle_faces = []
            for face in reduced_faces:
                for j in range(1, len(face) - 1):
                    reduced_triangle_faces.append([face[0],
                                                    face[j],
                                                    face[j+1]])
            mesh = trimesh.Trimesh(vertices=reduced_vertices, faces=reduced_triangle_faces)
            mesh.fix_normals()
            mesh.fill_holes()
            mesh.remove_degenerate_faces()
            mesh.remove_unreferenced_vertices()
            if not validate_mesh(mesh):
                return None
            if min(count_elements(reduced_faces).values())<3:
                return None
            final_vertices, final_faces = np.array(mesh.vertices), np.array(mesh.faces)
        except:
            return None
        pieces[f"frag{k}"]={"faces":[list([int(f) for f in face]) for face in final_faces if len(face)!=0], "vertices":[list(vert) for vert in final_vertices]}

    return pieces

from collections import Counter

def count_elements(nested_list):
    # Flatten the nested list to a single list
    flattened_list = [element for sublist in nested_list for element in sublist]
    
    # Use Counter to count occurrences of each element
    element_counts = Counter(flattened_list)
    
    return element_counts

def generate_randomcut3d(split = 'train', piecemin=7, piecemax=7, num_initial_points=7):
    if split == 'train':
        num_shape = 5000
        assert num_shape % (piecemax - piecemin + 1) == 0, f"{num_shape} is not divisible by {(piecemax - piecemin + 1)}"
        num_shape_per_piece = num_shape // (piecemax - piecemin + 1)
    if split == 'test':
        num_shape = 50
        assert num_shape % (piecemax - piecemin + 1) == 0, f"{num_shape} is not divisible by {(piecemax - piecemin + 1)}"
        num_shape_per_piece = num_shape // (piecemax - piecemin + 1)
    
    datacnt=0

    base_dir = f"./datasets/randomcut_3d/{split}_data"
    os.makedirs(base_dir, exist_ok=True)

    for num_piece in range(piecemin, piecemax + 1, 1):
        piece_dir = os.path.join(base_dir, f"{num_piece}pieces")
        os.makedirs(piece_dir, exist_ok=True)
        for num_data in tqdm(range(num_shape_per_piece)):
            succeed = False
            while not succeed:
                # 전체 모형 생성 (set of vertices)
                points = generate_random_points(num_points=num_initial_points)
                hull = create_convex_hull(points)
                idx_mapping = np.empty(num_initial_points, dtype=int)
                for i, v in enumerate(hull.vertices):
                    idx_mapping[v] = int(i)
                vertices = points[hull.vertices]
                faces = idx_mapping[hull.simplices]
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

                # 모형을 자르기
                sliced_meshes = slice_mesh(mesh, piece_num=num_piece)

                # 자른 모형 일부 합치기
                pieces = reduce_meshes(sliced_meshes)
                if pieces == None:
                    continue
                else:
                    succeed = True
                    with open (os.path.join(piece_dir, f"puzzle_{num_data}.json"), "w") as f:
                        json.dump(pieces, f)
            
if __name__ == "__main__":
    generate_randomcut3d('train')
    generate_randomcut3d('test')