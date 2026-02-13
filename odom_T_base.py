import numpy as np
import rosbag
import transformations as tr
import json

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
    for topic, msg, _ in bag.read_messages(topics=["/tf"]):
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
    best = None

    while lo <= hi:
        mid = (lo + hi) // 2
        t_mid = tf_list[mid][0]

        if t_mid <= stamp:
            best = tf_list[mid][1]
            lo = mid + 1
        else:
            hi = mid - 1
    if abs(stamp - t_mid) <= 0.02:
        return best  # (stamp, matrix) 或 None
    return None

# bag_path = '/home/gef/gef/data/2022-02-18-09-39-16_0.bag' # 166 / 3831
bag_path = '/home/gef/gef/data/2022-02-18-09-53-36_38.bag' # 61 / 3831
bag_path = '/home/gef/gef/data/2022-01-20-16-34-01_0.bag' # 440 / 3831

json_path = "/home/gef/gef/data/2022-02-18/scene_info/samples_scenes.json"
import datetime

with open(json_path, "r") as f:
    data = json.load(f)

samples = data["samples"]  # 假设结构是这样
with rosbag.Bag(bag_path, 'r') as bag:
    tf_list = build_odom_base_tf(bag, base_frame="base_footprint")
    print("共读取 TF 条数:", len(tf_list))

valid_count = 0
a = []
for _, sample in samples.items():
    stamp = float(sample["timestamp"])
    dt = datetime.datetime.fromtimestamp(stamp)
    a.append(dt)
    print(dt) 

    odom_T_base = lookup_odom_T_base(tf_list, stamp)
    if odom_T_base is None:
        sample["odom_T_base"] = None
        continue
    # numpy → list（JSON 不能存 numpy）
    sample["odom_T_base"] = odom_T_base.tolist()
    valid_count += 1

print(f"Attached odom_T_base to {valid_count}/{len(samples)} samples")
out_path = json_path.replace(".json", "_with_odom.json")

with open(out_path, "w") as f:
    json.dump(data, f, indent=2)

print("Saved:", out_path)
