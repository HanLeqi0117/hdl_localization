#include <mutex>
#include <memory>
#include <iostream>

#include <rclcpp/rclcpp.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_ros/transform_broadcaster.h>

#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl/filters/voxel_grid.h>

namespace hdl_localization {

class GlobalmapServerNode : public rclcpp::Node {
    public:
        using PointT = pcl::PointXYZI;

        GlobalmapServerNode(const std::string node_name, const rclcpp::NodeOptions options) : Node(node_name, options){

            // Get the Point Cloud Data from a pcd file and downsample the Point Cloud Data.
            // If the Point Cloud Map is made in UTM Environment, Change the UTM coordinate into Local coordinate
            initialize_parameters();

            // publish globalmap with "latched" publisher
            // auto global_map_qos = rclcpp::SystemDefaultsQoS();
            // global_map_qos.get_rmw_qos_profile().depth = 5;
            // global_map_qos.get_rmw_qos_profile().reliability = RMW_QOS_POLICY_RELIABILITY_RELIABLE;
            // global_map_qos.get_rmw_qos_profile().durability = RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL;
            auto global_map_qos = rclcpp::QoS(1).transient_local();
            globalmap_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("global_map", global_map_qos);
            // When get a PointCloud Data, publish it as a new PointCloud Map.
            map_update_sub = this->create_subscription<std_msgs::msg::String>("map_request/pcd", 1, std::bind(&GlobalmapServerNode::map_update_callback, this, std::placeholders::_1));
            // Publish the Point Cloud Map which was downsampled.
            globalmap_pub_timer = this->create_wall_timer(std::chrono::seconds(1), std::bind(&GlobalmapServerNode::pub_once_cb, this));

        }
        virtual ~GlobalmapServerNode(){}

    private:
        void initialize_parameters(){
            // read globalmap from a pcd file
            std::string globalmap_pcd = this->declare_parameter<std::string>("globalmap_pcd", "");
            globalmap.reset(new pcl::PointCloud<PointT>());
            pcl::io::loadPCDFile(globalmap_pcd, *globalmap);
            globalmap->header.frame_id = "map";

            std::ifstream utm_file(globalmap_pcd + ".utm");
            if (utm_file.is_open() && this->declare_parameter<bool>("convert_utm_to_local", true)) {
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
            double downsample_resolution = this->declare_parameter<double>("downsample_resolution", 0.1);
            boost::shared_ptr<pcl::VoxelGrid<PointT>> voxelgrid(new pcl::VoxelGrid<PointT>());
            voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
            voxelgrid->setInputCloud(globalmap);

            pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
            voxelgrid->filter(*filtered);

            globalmap = filtered;
        }
    
        void map_update_callback(std_msgs::msg::String::ConstSharedPtr msg){
            RCLCPP_INFO_STREAM(get_logger(), "Received map request, map path : "<< msg->data);
            std::string globalmap_pcd = msg->data;
            globalmap.reset(new pcl::PointCloud<PointT>());
            pcl::io::loadPCDFile(globalmap_pcd, *globalmap);
            globalmap->header.frame_id = "map";

            // downsample globalmap
            double downsample_resolution = this->declare_parameter<double>("downsample_resolution", 0.1);
            boost::shared_ptr<pcl::VoxelGrid<PointT>> voxelgrid(new pcl::VoxelGrid<PointT>());
            voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
            voxelgrid->setInputCloud(globalmap);

            pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
            voxelgrid->filter(*filtered);

            globalmap = filtered;
            // globalmap_pub->publish(globalmap);
            sensor_msgs::msg::PointCloud2 globalmap_msg;
            pcl::toROSMsg(*globalmap, globalmap_msg);            
            
            globalmap_pub->publish(globalmap_msg);

        }

        void pub_once_cb() {
            sensor_msgs::msg::PointCloud2 globalmap_msg;
            pcl::toROSMsg(*globalmap, globalmap_msg);            
            globalmap_pub->publish(globalmap_msg);
            globalmap_pub_timer.reset();
        }
        
    private:

        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr globalmap_pub;
        rclcpp::Subscription<std_msgs::msg::String>::SharedPtr map_update_sub;
        rclcpp::TimerBase::SharedPtr globalmap_pub_timer;

        pcl::PointCloud<PointT>::Ptr globalmap;
};

}

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    options.automatically_declare_parameters_from_overrides(false);
    rclcpp::spin(std::make_shared<hdl_localization::GlobalmapServerNode>("globalmap_server_node", options));

    rclcpp::shutdown();
    return 0;
}
