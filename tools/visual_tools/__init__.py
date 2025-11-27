import open3d as o3d
import numpy as np
import os

from .open3d_coordinate import create_coordinate
from .open3d_arrow import create_arrow
from .open3d_box import create_box


def create_box_with_arrow(box, box_color=None, arrow_color=None):
    """
    box: list(8) [ x, y, z, dx, dy, dz, yaw]
    """

    box_o3d = create_box(box, box_color)
    x = box[0]
    y = box[1]
    z = box[2]
    l = box[3]
    yaw = box[6]
    # get direction arrow
    dir_x = l / 2.0 * np.cos(yaw)
    dir_y = l / 2.0 * np.sin(yaw)

    arrow_origin = [x - dir_x, y - dir_y, z]
    arrow_end = [x + dir_x, y + dir_y, z]
    arrow = create_arrow(arrow_origin, arrow_end, arrow_color)

    return box_o3d, arrow

def draw_clouds_with_boxes(cloud, boxes, save_path="output.png"):
    """
    cloud: (N, 4)  [x, y, z, intensity]
    boxes: (n, 7)  [x, y, z, dx, dy, dz, yaw]
    """

    # --------------------------------------------------------------
    # 判断是否存在图形显示环境（DISPLAY）
    # --------------------------------------------------------------
    headless = not os.environ.get("DISPLAY")

    if headless:
        o3d.visualization.webrtc_server.enable_webrtc()
        print("⚙️ 无 DISPLAY 环境，启用离屏渲染模式 (OffscreenRenderer)")
        renderer = o3d.visualization.rendering.OffscreenRenderer(1280, 720)
        print(1)

        scene = renderer.scene
        print(1)
        # 创建点云
        points_color = np.tile([[0.5, 0.5, 0.5]], (cloud.shape[0], 1))
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(cloud[:, :3])
        pc.colors = o3d.utility.Vector3dVector(points_color)

        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        scene.add_geometry("pointcloud", pc, mat)

        # 添加 box（如果有）
        if boxes is not None and len(boxes) > 0:
            for i, box in enumerate(boxes):
                # 根据 box 参数生成 3D 框
                bbox = o3d.geometry.OrientedBoundingBox(
                    center=box[:3],
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, box[6]]),
                    extent=box[3:6],
                )
                bbox.color = (1, 0, 0)
                scene.add_geometry(f"box_{i}", bbox, mat)

        # 渲染到图像并保存
        img = renderer.render_to_image()
        o3d.io.write_image(save_path, img)
        print(f"✅ 离屏渲染完成，已保存图像到：{save_path}")

    else:
        print("🖥️ 检测到 DISPLAY，使用窗口可视化模式")
        vis = o3d.visualization.Visualizer()

        vis.create_window()
        # --------------------------------------------------------------
        # create point cloud
        # --------------------------------------------------------------
        points_color = [[1, 1, 1]]  * cloud.shape[0]
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(cloud[:,:3])
        pc.colors = o3d.utility.Vector3dVector(points_color)
        vis.add_geometry(pc)

        # --------------------------------------------------------------
        # create boxes with colors with arrow
        # --------------------------------------------------------------
        boxes_o3d = []

        cur_box_color = [0, 1, 0]
        cur_arrow_color = [1, 0, 0]
        # create boxes
        for box in boxes:
            box_o3d, arrow = create_box_with_arrow(box, cur_box_color, cur_arrow_color)
            boxes_o3d.append(box_o3d)
            # boxes_o3d.append(arrow)
        # add_geometry fro boxes
        [vis.add_geometry(element) for element in boxes_o3d]

        # --------------------------------------------------------------
        # coordinate frame
        # --------------------------------------------------------------
        coordinate_frame = create_coordinate(size=2.0, origin=[0, 0, 0])
        vis.add_geometry(coordinate_frame)

        # --------------------------------------------------------------
        # drop the window
        # --------------------------------------------------------------
        vis.get_render_option().point_size = 2
        vis.get_render_option().background_color = np.array([0.0, 0.0, 0.0])

        vis.run()
        vis.destroy_window()

