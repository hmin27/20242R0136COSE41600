import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser(description="Process PCD files.")
parser.add_argument('folder_path', type=str, help='Path to the folder containing PCD files')
args = parser.parse_args()

# PCD 데이터 로드
folder_path = args.folder_path
file_names = sorted([f for f in os.listdir(folder_path) if f.endswith('.pcd')])
file_index = 0

# VisualizerWithKeyCallback 생성 및 창 열기
visualizer = o3d.visualization.VisualizerWithKeyCallback()
visualizer.create_window(window_name="Filtered Clusters and Bounding Boxes", width=720, height=720)


# 카메라 뷰포인트 저장 변수
view_control = visualizer.get_view_control()
current_view_params = None

# 클러스터링 및 필터링을 수행하는 함수
def process_pcd(file_path):
    global cluster_colors
    original_pcd = o3d.io.read_point_cloud(file_path)

    # Voxel Downsampling 수행
    voxel_size = 0.2
    downsample_pcd = original_pcd.voxel_down_sample(voxel_size=voxel_size)

    # Radius Outlier Removal (ROR) 적용
    cl, ind = downsample_pcd.remove_radius_outlier(nb_points=6, radius=1.2)
    ror_pcd = downsample_pcd.select_by_index(ind)

    # RANSAC을 사용하여 평면 제거
    plane_model, inliers = ror_pcd.segment_plane(distance_threshold=0.1,
                                                 ransac_n=3,
                                                 num_iterations=2000)
    final_point = ror_pcd.select_by_index(inliers, invert=True)

    # 포인트 클라우드를 NumPy 배열로 변환
    points = np.asarray(final_point.points)

    # DBSCAN 클러스터링 적용
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(final_point.cluster_dbscan(eps=0.3, min_points=10, print_progress=True))

    max_label = labels.max()
    print(f"{file_path} point cloud has {max_label + 1} clusters")

    # 노이즈를 제거하고 각 클러스터에 색상 지정
    colors = plt.get_cmap("tab20")(labels / (max_label + 1 if max_label > 0 else 1))
    colors[labels < 0] = 0  # 노이즈는 검정색으로 표시
    final_point.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # 필터링 기준 1. 클러스터 내 최대 최소 포인트 수
    min_points_in_cluster = 5
    max_points_in_cluster = 40

    # 필터링 기준 2. 클러스터 내 최소 최대 Z값
    min_z_value = -1.5
    max_z_value = 2.5

    # 필터링 기준 3. 클러스터 내 최소 최대 Z값 차이
    min_height = 0.5
    max_height = 2.0

    max_distance = 30.0

    # 1번, 2번, 3번 조건을 모두 만족하는 클러스터 필터링 및 바운딩 박스 생성
    bboxes_1234 = []
    for i in range(max_label + 1):
        cluster_indices = np.where(labels == i)[0]
        if min_points_in_cluster <= len(cluster_indices) <= max_points_in_cluster:
            cluster_pcd = final_point.select_by_index(cluster_indices)
            points = np.asarray(cluster_pcd.points)
            z_values = points[:, 2]
            z_min = z_values.min()
            z_max = z_values.max()
            if min_z_value <= z_min and z_max <= max_z_value:
                height_diff = z_max - z_min
                if min_height <= height_diff <= max_height:
                    distances = np.linalg.norm(points, axis=1)
                    if distances.max() <= max_distance:
                        bbox = cluster_pcd.get_axis_aligned_bounding_box()
                        bbox.color = (1, 0, 0)
                        bboxes_1234.append(bbox)

    return final_point, bboxes_1234


# Callback Function 정의
def update_pcd(vis, index, view_control, viewpoint_params):
    vis.clear_geometries()
    file_path = os.path.join(folder_path, file_names[index])
    
    final_point, bboxes_1234 = process_pcd(file_path)
    vis.add_geometry(final_point)
    for bbox in bboxes_1234:
        vis.add_geometry(bbox)
    
    vis.get_render_option().point_size = 2.0

    if viewpoint_params is not None:
        view_control.convert_from_pinhole_camera_parameters(viewpoint_params, allow_arbitrary=True)
         
def load_next_pcd(vis):
    global file_index, current_view_params
    
    current_view_params = view_control.convert_to_pinhole_camera_parameters()

    if file_index < len(file_names) - 1:
        file_index += 1
        update_pcd(vis, file_index, view_control, current_view_params)
 
    return False

def load_previous_pcd(vis):
    global file_index, current_view_params

    current_view_params = view_control.convert_to_pinhole_camera_parameters()

    if file_index > 0:
        file_index -= 1
        update_pcd(vis, file_index, view_control, current_view_params)

    return False

def quit_visualizer(vis):
    vis.destroy_window()
    return False


# 첫 번째 파일 로드
update_pcd(visualizer, file_index, view_control, current_view_params)
current_view_params = view_control.convert_to_pinhole_camera_parameters()

visualizer.register_key_callback(ord("N"), load_next_pcd)
visualizer.register_key_callback(ord("P"), load_previous_pcd)
visualizer.register_key_callback(ord("Q"), quit_visualizer)

visualizer.run()
visualizer.destroy_window()