#include <iostream>
#include "det3d_publisher.hpp"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "pc_infer");
  Det3DPublisher app;
  ros::spin();
  return 0;
}