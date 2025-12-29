#include "det3d_publisher.hpp"

#include <std_msgs/Header.h>

namespace {
    constexpr char kDefaultCloudTopic[] = "/cartographer_ros/merge_point_cloud";
    constexpr char kDefaultObjectsTopic[] = "/detection/lidar_detector/objects3d";
    constexpr char kDefaultPfePath[] =
        "/home/gef/catkin_3d/src/PointPillars_MultiHead_40FPS/model/jz_pfe_1.8.trt";
    constexpr char kDefaultBackbonePath[] =
        "/home/gef/catkin_3d/src/PointPillars_MultiHead_40FPS/model/jz_backbone_1.8.trt";
    constexpr char kDefaultConfigPath[] =
        "/home/gef/catkin_3d/src/PointPillars_MultiHead_40FPS/pointpillars/cfgs/pointpillars_hv_fpn_sbn-all_8xb4-2x_jz-3d.yaml";
}  // namespace

Det3DPublisher::Det3DPublisher()
    : cloud_topic_(kDefaultCloudTopic),
      objects_topic_(kDefaultObjectsTopic),
      pfe_path_(kDefaultPfePath),
      backbone_path_(kDefaultBackbonePath),
      config_path_(kDefaultConfigPath){
    nh_.param("pointpillars/cloud_topic", cloud_topic_, cloud_topic_);
    nh_.param("pointpillars/objects_topic", objects_topic_, objects_topic_);
    nh_.param("pointpillars/pfe_path", pfe_path_, pfe_path_);
    nh_.param("pointpillars/backbone_path", backbone_path_, backbone_path_);
    nh_.param("pointpillars/config_path", config_path_, config_path_);
    nh_.param("pointpillars/use_onnx", use_onnx_, use_onnx_);
    nh_.param("pointpillars/score_threshold", score_threshold_, score_threshold_);
    nh_.param("pointpillars/nms_threshold", nms_threshold_, nms_threshold_);

    loadClassNames();

    ROS_INFO_STREAM("PointPillars params:"
                    << "\n  \x1b[32m pfe_path: \x1b[0m" << pfe_path_
                    << "\n  \x1b[32m backbone_path: \x1b[0m" << backbone_path_
                    << "\n  \x1b[32m config_path: \x1b[0m" << config_path_
                    << "\n  \x1b[32m use_onnx: \x1b[0m" << (use_onnx_ ? "true" : "false")
                    << "\n  \x1b[32m score_threshold: \x1b[0m" << score_threshold_
                    << "\n  \x1b[32m nms_threshold: \x1b[0m" << nms_threshold_
                    << "\n  \x1b[32m cloud_topic: \x1b[0m" << cloud_topic_
                    << "\n  \x1b[32m objects_topic: \x1b[0m" << objects_topic_);

    detector_.reset(new PointPillars(
        score_threshold_, nms_threshold_, use_onnx_,
        pfe_path_, backbone_path_, config_path_));

    objects_pub_ = nh_.advertise<autoware_msgs::DetectedObjectArray>(objects_topic_, 1);
    cloud_sub_ = nh_.subscribe(cloud_topic_, 1, &Det3DPublisher::pointCloudCallback, this);
}

void Det3DPublisher::loadClassNames() {
    try {
        YAML::Node params = YAML::LoadFile(config_path_);
        if (params["CLASS_NAMES"]) {
            class_names_ = params["CLASS_NAMES"].as<std::vector<std::string>>();
        }
    } catch (const std::exception& e) {
        ROS_WARN_STREAM("读取类别名失败: " << e.what());
    }

    if (class_names_.empty()) {
        class_names_ = {"car", "truck", "trailer", "bus", "construction_vehicle",
                        "bicycle", "motorcycle", "pedestrian",
                        "traffic_cone", "barrier"};
        ROS_WARN("使用默认类别列表");
    }
}

void Det3DPublisher::pointCloudCallback(
    const sensor_msgs::PointCloud2ConstPtr& msg) {
      if (!detector_) {
        ROS_ERROR_THROTTLE(1.0, "PointPillars 未初始化");
        return;
    }

    const size_t num_points = static_cast<size_t>(msg->width) * msg->height;
    if (num_points == 0) {
        return;
    }

    std::vector<float> points;
    points.reserve(num_points * 4);

    try {
        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");
        sensor_msgs::PointCloud2ConstIterator<float> iter_intensity(*msg, "intensity");

        for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z, ++iter_intensity) {
            points.emplace_back(*iter_x);
            points.emplace_back(*iter_y);
            points.emplace_back(*iter_z);
            points.emplace_back(*iter_intensity);
        }
    } catch (const std::exception& e) {
        ROS_ERROR_STREAM_THROTTLE(1.0, "解析点云失败: " << e.what());
        return;
    }

    const auto boxes = detector_->DoInference(
        points.data(), static_cast<int>(num_points));
    auto objects_msg = convertToMsg(boxes, msg->header);
    objects_pub_.publish(objects_msg);
}

autoware_msgs::DetectedObjectArray Det3DPublisher::convertToMsg(
    const std::vector<BoundingBox>& boxes, const std_msgs::Header& header) const {
    autoware_msgs::DetectedObjectArray array_msg;
    array_msg.header = header;
    array_msg.objects.reserve(boxes.size());

    for (const auto& box : boxes) {
        autoware_msgs::DetectedObject obj;
        obj.header = header;
        obj.id = box.id;
        obj.score = box.score;
        obj.label = (box.id >= 0 && static_cast<size_t>(box.id) < class_names_.size())
                        ? class_names_[static_cast<size_t>(box.id)]
                        : "unknown";

        obj.pose.position.x = box.x;
        obj.pose.position.y = box.y;
        obj.pose.position.z = box.z;
        
        // TODO LEARN
        float yaw = box.rt;
        // yaw += M_PI / 2;
        yaw = std::atan2(std::sin(yaw), std::cos(yaw));    
        obj.pose.orientation = tf::createQuaternionMsgFromYaw(yaw); 

        obj.dimensions.x = box.l;
        obj.dimensions.y = box.w;
        obj.dimensions.z = box.h;

        obj.valid = true;
        obj.pose_reliable = true;
        obj.velocity_reliable = false;
        obj.acceleration_reliable = false;
        obj.velocity.linear.x = 0.0;
        obj.velocity.linear.y = 0.0;
        obj.velocity.linear.z = 0.0;

        array_msg.objects.emplace_back(obj);
    }

    return array_msg;
}
