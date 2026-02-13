import pointpillars_py
import numpy as np
import cv2
import time
import rosbag
import sensor_msgs.point_cloud2 as pc2
import transformations as tr

def save_bev(cloud, boxes, save_path="bev.png",
                 res=0.1,          # BEV 分辨率: 1 pixel = 0.05m
                 xrange=(-50, 50),  # x 范围
                 yrange=(-50, 50)): # y 范围

    # 创建空画布（黑色）
    H = int((yrange[1] - yrange[0]) / res)
    W = int((xrange[1] - xrange[0]) / res)
    bev = np.zeros((H, W, 3), dtype=np.uint8)

    # 坐标转换到像素 (BEV 以前方为上: y 轴反转)
    xs = ((cloud[:, 0] - xrange[0]) / res).astype(np.int32)
    ys = ((yrange[1] - cloud[:, 1]) / res).astype(np.int32)

    mask = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    bev[ys[mask], xs[mask]] = (255, 255, 255)  # 白色点云

    # 绘制 box (box在点云所在的的坐标系)
    if boxes is not None:
        for box in boxes:
            x, y, dx, dy, yaw = box.x, box.y, box.w, box.l, box.rt

            # 四角坐标
            corners = np.array([
                [ dx/2,  dy/2],
                [ dx/2, -dy/2],
                [-dx/2, -dy/2],
                [-dx/2,  dy/2],
            ])
            rot = np.array([[np.cos(yaw), -np.sin(yaw)],
                            [np.sin(yaw),  np.cos(yaw)]])
            corners = (corners @ rot.T) + np.array([x, y])

            # 转像素坐标
            pts = np.zeros((4, 2), dtype=np.int32)
            pts[:, 0] = ((corners[:, 0] - xrange[0]) / res).astype(np.int32)
            pts[:, 1] = ((yrange[1] - corners[:, 1]) / res).astype(np.int32)

            cv2.polylines(bev, [pts], True, (0, 255, 0), 2) 
    return bev

def tf_to_matrix(tf):
    q = [
        tf.transform.rotation.x,
        tf.transform.rotation.y,
        tf.transform.rotation.z,
        tf.transform.rotation.w
    ]
    t = [
        tf.transform.translation.x,
        tf.transform.translation.y,
        tf.transform.translation.z
    ]
    M = tr.quaternion_matrix(q)
    M[:3, 3] = t
    return M

def build_odom_base_tf(bag, base_frame="base_footprint"):
    tf_list = []  # [(stamp, odom_T_base), ...]

    # 从 /tf + /tf_static 里找 odom->base_footprint
    for _, msg, _ in bag.read_messages(topics=["/tf", "/tf_static"]):
        for tf in msg.transforms:
            parent = tf.header.frame_id
            child  = tf.child_frame_id

            # 只收集 odom → base_footprint
            if parent == "odom" and child == base_frame:
                stamp = tf.header.stamp.to_sec()
                tf_list.append((stamp, tf_to_matrix(tf)))

    # 按时间排序
    tf_list.sort(key=lambda x: x[0])
    return tf_list

def lookup_odom_T_base(tf_list, stamp):
    """
    返回最接近 stamp 的 (t_found, odom_T_base)
    且 t_found <= stamp
    """
    lo = 0
    hi = len(tf_list) - 1
    best = np.eye(4)

    while lo <= hi:
        mid = (lo + hi) // 2
        t_mid = tf_list[mid][0]

        if t_mid <= stamp:
            best = tf_list[mid][1]
            lo = mid + 1
        else:
            hi = mid - 1

    return best  # (stamp, matrix) 或 None

score_threshold = 0.3
nms_overlap_threshold = 0.2
use_onnx_bool = False
# pfe_file = '/home/gef/catkin_3d/src/PointPillars_MultiHead_40FPS/model/jz_vfe_bchw.trt'
# rpn_file = '/home/gef/catkin_3d/src/PointPillars_MultiHead_40FPS/model/jz_backbone.trt'
# cfg_yaml_path = '/home/gef/catkin_3d/src/PointPillars_MultiHead_40FPS/pointpillars/cfgs/pointpillars_hv_fpn_sbn-all_8xb4-2x_jz-3d.yaml'

pfe_file = '/home/gef/catkin_3d/src/PointPillars_MultiHead_40FPS/model/test_bchw.trt'
# pfe_file = '/home/gef/catkin_3d/src/PointPillars_MultiHead_40FPS/model/test_pfe.trt'
rpn_file = '/home/gef/catkin_3d/src/PointPillars_MultiHead_40FPS/model/test_backbone.trt'
cfg_yaml_path = '/home/gef/catkin_3d/src/PointPillars_MultiHead_40FPS/pointpillars/cfgs/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.yaml'

# 初始化（参数与C++构造一致）
pp = pointpillars_py.PointPillars(
    score_threshold,
    nms_overlap_threshold,
    use_onnx_bool,
    pfe_file,
    rpn_file,
    cfg_yaml_path
)
output_video = 'bev_video.mp4'
frame_width, frame_height = 1000, 1000  # 根据 save_bev 输出大小调整
fps = 10
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

bag_path = '/home/gef/catkin_3d/src/PointPillars_MultiHead_40FPS/2022-03-08-09-41-57.bag'
topic_name = '/cartographer_ros/merge_point_cloud' # ego坐标系，地面为0
# n_sweeps = 10
max_n_points = 200000
points = np.zeros((max_n_points, 4))
# 循环0 —— max_n_points 来填与预分配的array
i = 0
ts_i = 0
with rosbag.Bag(bag_path, 'r') as bag:
    tf_list = build_odom_base_tf(bag, base_frame="base_footprint")
    print("共读取 TF 条数:", len(tf_list))
    pre_t = next(bag.read_messages(topics=[topic_name]))[2] 
    odom_T_prebase = np.eye(4)
    for topic, msg, cur_t in bag.read_messages(topics=[topic_name]):
        if ts_i  == 1000:
            break
        ts_i += 1
        t1 = time.time()
        ts_val = cur_t.to_time() - pre_t.to_time()
        points[..., 3] += ts_val
        odom_T_curbase = lookup_odom_T_base(tf_list, cur_t.to_time())
        curbase_T_prebase = np.linalg.inv(odom_T_curbase) @ odom_T_prebase
        R = curbase_T_prebase[:3, :3]
        t = curbase_T_prebase[:3, 3]
        points[..., :3] = points[..., :3] @ R.T + t
        # t1 = time.time()
        data = np.frombuffer(msg.data, dtype=np.uint8)
        data = data.reshape((-1, msg.point_step))
        x = data[:, 0:4].view(np.float32).reshape(-1)
        y = data[:, 4:8].view(np.float32).reshape(-1)
        z = data[:, 8:12].view(np.float32).reshape(-1)
        ts = np.zeros(z.shape)
        cur_points = np.stack([x, y, z, ts], axis=1)
        num_pts = cur_points.shape[0]
        start = i
        end = i + num_pts
        if end <= max_n_points:
            # 不越界，直接写
            points[start:end] = cur_points
        else:
            # 越界 → 分两段写
            part1 = max_n_points - start        # 写到数组尾部
            points[start:] = cur_points[:part1]  # 第一段
            points[:end - max_n_points] = cur_points[part1:]  # 第二段
        # points = np.fromfile("/home/gef/catkin_3d/src/PointPillars_MultiHead_40FPS/tmp/test.bin", dtype=np.float32).reshape(-1, 4) # 雷达坐标系，地面点为-1.8左右
        print(f"数据读取{time.time() - t1}")
        boxes = pp.DoInference(points) # 输入点云各维度应该为(x, y, z, cur_ts-sweeps_ts)
        bev = save_bev(points, boxes)
        bev_resized = cv2.resize(bev, (frame_width, frame_height))
        cv2.imwrite("bev.png", bev_resized)
        out.write(bev_resized)
        pre_t = cur_t
        odom_T_prebase = odom_T_curbase

        i += num_pts
        i = i % max_n_points

out.release()
print(f"✅ 视频已保存到 {output_video}")

# points2 = np.fromfile("/data/gef/PointPillars_MultiHead_40FPS/tmp/test.bin", dtype=np.float32).reshape(-1, 4)

    

