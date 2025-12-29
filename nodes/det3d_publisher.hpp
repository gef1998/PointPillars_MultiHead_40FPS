#ifndef DET3D_PUBLISHER_HPP
#define DET3D_PUBLISHER_HPP

#include <memory>
#include <string>
#include <vector>

#include <autoware_msgs/DetectedObjectArray.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <std_msgs/Header.h>
#include <tf/transform_datatypes.h>
#include <yaml-cpp/yaml.h>

#include "bbox.h"
#include "pointpillars.h"

/**
 * @brief PointPillars 3D 目标检测 ROS 发布器
 */
class Det3DPublisher {
public:
    Det3DPublisher();

private:
    void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg);
    autoware_msgs::DetectedObjectArray convertToMsg(
        const std::vector<BoundingBox>& boxes, const std_msgs::Header& header) const;
    void loadClassNames();

    ros::NodeHandle nh_;
    ros::Subscriber cloud_sub_;
    ros::Publisher objects_pub_;

    std::unique_ptr<PointPillars> detector_;

    std::string cloud_topic_;
    std::string objects_topic_;
    std::string pfe_path_;
    std::string backbone_path_;
    std::string config_path_;
    bool use_onnx_{false};
    float score_threshold_{0.25f};
    float nms_threshold_{0.2f};

    std::vector<std::string> class_names_;
};

#endif // DET3D_PUBLISHER_HPP

