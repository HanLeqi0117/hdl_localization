#include <mutex>
#include <memory>
#include <iostream>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

namespace hdl_localization {

class GlobalmapServerNodelet : public rclcpp::Node {
public:
  using PointT = pcl::PointXYZI;

  GlobalmapServerNodelet(const rclcpp::NodeOptions& options)
  : Node("map_server", options)
  {
    initialize_params();

    // publish globalmap with "latched" publisher
    auto latch_qos = rclcpp::QoS(1).transient_local();
    globalmap_pub = create_publisher<sensor_msgs::msg::PointCloud2>("/globalmap", latch_qos);
    globalmap_pub_timer = create_wall_timer(std::chrono::milliseconds(1000), std::bind(&GlobalmapServerNodelet::pub_once_cb, this));
  }

private:
  void initialize_params() {
    // read globalmap from a pcd file
    std::string globalmap_pcd = declare_parameter<std::string>("globalmap_pcd", std::string(""));
    globalmap.reset(new pcl::PointCloud<PointT>());
    pcl::io::loadPCDFile(globalmap_pcd, *globalmap);
    globalmap->header.frame_id = "map";

    bool convert_utm_to_local = declare_parameter<bool>("convert_utm_to_local", true);
    std::ifstream utm_file(globalmap_pcd + ".utm");
    if (utm_file.is_open() && convert_utm_to_local) {
      double utm_easting;
      double utm_northing;
      double altitude;
      utm_file >> utm_easting >> utm_northing >> altitude;
      for(auto& pt : globalmap->points) {
        pt.getVector3fMap() -= Eigen::Vector3f(utm_easting, utm_northing, altitude);
      }
      RCLCPP_INFO_STREAM(get_logger(), "Global map offset by UTM reference coordinates (x = "
                      << utm_easting << ", y = " << utm_northing << ") and altitude (z = " << altitude << ")");
    }

    // downsample globalmap
    double downsample_resolution = declare_parameter<double>("downsample_resolution", 0.1);
    boost::shared_ptr<pcl::VoxelGrid<PointT>> voxelgrid(new pcl::VoxelGrid<PointT>());
    voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
    voxelgrid->setInputCloud(globalmap);

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    voxelgrid->filter(*filtered);

    globalmap = filtered;
  }

  void pub_once_cb() {
    sensor_msgs::msg::PointCloud2 globalmap_msg;
    pcl::toROSMsg(*globalmap, globalmap_msg);
    globalmap_pub->publish(globalmap_msg);
    globalmap_pub_timer.reset();
  }

private:
  // ROS
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr globalmap_pub;

  rclcpp::TimerBase::SharedPtr globalmap_pub_timer;
  pcl::PointCloud<PointT>::Ptr globalmap;
};

}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(hdl_localization::GlobalmapServerNodelet)
