import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy
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

    # Radius Outlier Removal (ROR) 적용
    cl, ind = downsample_pcd.remove_radius_outlier(nb_points=6, radius=1.2)
    ror_pcd = downsample_pcd.select_by_index(ind)

    # RANSAC을 사용하여 평면 제거
    plane_model, inliers = ror_pcd.segment_plane(distance_threshold=0.1,
                                                 ransac_n=3,
                                                 num_iterations=2000)
    final_point = ror_pcd.select_by_index(inliers, invert=True)

    # DBSCAN 클러스터링 적용
    labels = np.array(final_point.cluster_dbscan(eps=0.6, min_points=10, print_progress=True))
    colors = np.tile([0, 0, 1], (len(labels), 1))  # 파란색
    colors[labels < 0] = 0  # 노이즈는 검정색
    final_point.colors = o3d.utility.Vector3dVector(colors[:, :3])

    return final_point, labels

# 이동한 객체를 ICP로 클러스터 단위로 검출하는 함수
def detect_moving_clusters_icp(previous_pcd, current_pcd, previous_centroids, current_centroids):
    # ICP registration
    icp_threshold = 0.5
    icp_result = o3d.pipelines.registration.registration_icp(
        source=current_pcd,
        target=previous_pcd,
        max_correspondence_distance=icp_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    transformation = icp_result.transformation

    # 현재 centroids 변환
    transformed_centroids = []
    for i, centroid in current_centroids:
        transformed_point = np.dot(transformation[:3, :3], centroid) + transformation[:3, 3]
        transformed_centroids.append((i, transformed_point))

    moving_clusters = []
    matched_clusters = set()

    # Matching
    for i_curr, transformed_centroid in transformed_centroids:
        min_distance = float('inf')
        matched_cluster = None
        for i_prev, centroid_prev in previous_centroids:
            distance = np.linalg.norm(transformed_centroid - centroid_prev)
            if distance < min_distance:
                min_distance = distance
                matched_cluster = i_prev
        if matched_cluster is not None and min_distance > 3.0:  # 이동 거리 임계값
            # 필터링 기준 적용
            cluster_height = transformed_centroid[2]
            cluster_height_diff = abs(transformed_centroid[2] - centroid_prev[2])

            # 클러스터 높이 범위와 높이 차이 조건
            if 1.4 <= cluster_height_diff <= 2.0 and cluster_height > 0:
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
    current_pcd, labels = process_pcd(file_path)

    centroids_curr = compute_centroids(current_pcd, labels)

    # 이동한 클러스터 검출
    if previous_centroids is not None and previous_pcd is not None:
        moving_cluster_ids = detect_moving_clusters_icp(previous_pcd, current_pcd, previous_centroids, centroids_curr)
        for i in moving_cluster_ids:
            cluster_indices = np.where(labels == i)[0]
            cluster_pcd = current_pcd.select_by_index(cluster_indices)
            bbox = cluster_pcd.get_axis_aligned_bounding_box()
            bbox.color = (1, 0, 0)
            vis.add_geometry(bbox)

    vis.add_geometry(current_pcd)
    vis.get_render_option().point_size = 2.0

    if viewpoint_params is not None:
        view_control.convert_from_pinhole_camera_parameters(viewpoint_params, allow_arbitrary=True)

    previous_pcd = copy.deepcopy(current_pcd)
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
