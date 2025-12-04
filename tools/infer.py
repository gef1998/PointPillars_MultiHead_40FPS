import pointpillars_py
import numpy as np
import cv2

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

    # 绘制 box
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

            cv2.polylines(bev, [pts], True, (0, 255, 0), 1)  # 红色框
    cv2.imwrite(save_path, bev)
    print(f"📌 BEV 图像已保存: {save_path}")

score_threshold = 0.1
nms_overlap_threshold = 0.2
use_onnx_bool = False
pfe_file = "/data/gef/PointPillars_MultiHead_40FPS/model/test_pfe.trt"
rpn_file = "/data/gef/PointPillars_MultiHead_40FPS/model/test_backbone.trt"
cfg_yaml_path = "/data/gef/PointPillars_MultiHead_40FPS/pointpillars/cfgs/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.yaml"


# 初始化（参数与C++构造一致）
pp = pointpillars_py.PointPillars(
    score_threshold,
    nms_overlap_threshold,
    use_onnx_bool,
    pfe_file,
    rpn_file,
    cfg_yaml_path
)

points = np.fromfile("/data/gef/PointPillars_MultiHead_40FPS/tmp/test.bin", dtype=np.float32).reshape(-1, 4)
in_num_points = len(points)# 点特征数（如5）

print("points shape:", np.array(points).shape)
print("in_num_points:", in_num_points)
print("points[:10]:", points[:10])  # 看首10个值

for _ in range(10):
    # 推理
    boxes, out_detections, out_labels, out_scores = pp.DoInference(points)
    save_bev(points, boxes)

