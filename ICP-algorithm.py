import open3d as o3d
import numpy as np
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

# 이전 PCD 저장 변수
previous_pcd = None
previous_centroids = None

# 클러스터링 및 필터링을 수행하는 함수
def process_pcd(file_path):
    original_pcd = o3d.io.read_point_cloud(file_path)

    # Voxel Downsampling 수행
    voxel_size = 0.2
    downsample_pcd = original_pcd.voxel_down_sample(voxel_size=voxel_size)

    # RANSAC을 사용하여 평면 추정 및 지면 제거
    plane_model, inliers = downsample_pcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=2000)
    non_ground_pcd = downsample_pcd.select_by_index(inliers, invert=True)

    # Radius Outlier Removal (ROR) 적용하여 노이즈 제거
    cl, ind = non_ground_pcd.remove_radius_outlier(nb_points=6, radius=1.2)
    filtered_pcd = non_ground_pcd.select_by_index(ind)

    return filtered_pcd

# 이동한 객체를 ICP로 클러스터 단위로 검출하는 함수
def detect_moving_clusters_icp(previous_centroids, current_centroids):
    moving_clusters = []
    matched_clusters = set()
    for i_curr, centroid_curr in current_centroids:
        min_distance = float('inf')
        matched_cluster = None
        for i_prev, centroid_prev in previous_centroids:
            distance = np.linalg.norm(centroid_curr - centroid_prev)
            if distance < min_distance:
                min_distance = distance
                matched_cluster = i_prev
        if matched_cluster is not None and min_distance > 5.0:  # 이동 거리 임계값
            matched_clusters.add(matched_cluster)
            moving_clusters.append(i_curr)

    return moving_clusters

# 각 클러스터의 중심 좌표 계산 함수
def compute_centroids(pcd, labels):
    centroids = []
    for i in range(labels.max() + 1):
        cluster_indices = np.where(labels == i)[0]
        cluster_points = pcd.select_by_index(cluster_indices)
        centroid = np.mean(np.asarray(cluster_points.points), axis=0)
        centroids.append((i, centroid))
    return centroids

# Callback Function 정의
def update_pcd(vis, index, view_control, viewpoint_params):
    global previous_pcd, previous_centroids

    vis.clear_geometries()
    file_path = os.path.join(folder_path, file_names[index])
    current_pcd = process_pcd(file_path)

    # 현재 PCD의 클러스터들을 검정색으로 표시
    labels = np.array(current_pcd.cluster_dbscan(eps=0.6, min_points=10, print_progress=True))
    max_label = labels.max()
    centroids_curr = compute_centroids(current_pcd, labels)

    # 이동한 클러스터 검출
    if previous_centroids is not None:
        moving_cluster_ids = detect_moving_clusters_icp(previous_centroids, centroids_curr)
        for i in range(max_label + 1):
            cluster_indices = np.where(labels == i)[0]
            cluster_pcd = current_pcd.select_by_index(cluster_indices)
            if i in moving_cluster_ids:
                cluster_pcd.paint_uniform_color([1, 0, 0])  # 이동한 클러스터는 빨간색으로 표시
            else:
                cluster_pcd.paint_uniform_color([0, 0, 0])  # 이동하지 않은 클러스터는 검정색으로 표시
            vis.add_geometry(cluster_pcd)
    else:
        for i in range(max_label + 1):
            cluster_indices = np.where(labels == i)[0]
            cluster_pcd = current_pcd.select_by_index(cluster_indices)
            cluster_pcd.paint_uniform_color([0, 0, 0])
            vis.add_geometry(cluster_pcd)

    vis.get_render_option().point_size = 2.0

    if viewpoint_params is not None:
        view_control.convert_from_pinhole_camera_parameters(viewpoint_params, allow_arbitrary=True)

    previous_pcd = current_pcd
    previous_centroids = centroids_curr

    del current_pcd

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

# 프로그램 종료 콜백 함수
def quit_visualizer(vis):
    vis.close()
    return False

# 첫 번째 파일 로드
update_pcd(visualizer, file_index, view_control, current_view_params)
current_view_params = view_control.convert_to_pinhole_camera_parameters()

visualizer.register_key_callback(ord("N"), load_next_pcd)
visualizer.register_key_callback(ord("P"), load_previous_pcd)
visualizer.register_key_callback(ord("Q"), quit_visualizer)

visualizer.run()
visualizer.destroy_window()