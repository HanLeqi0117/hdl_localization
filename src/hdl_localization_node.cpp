#include <mutex>
#include <memory>
#include <iostream>
#include <math.hpp>

#include <rclcpp/rclcpp.hpp>
#include <pcl_ros/point_cloud.hpp>
#include <pcl_ros/transforms.hpp>

#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
// #include <tf2_eigen_kdl/tf2_eigen_kdl.hpp>

#include <std_srvs/srv/empty.h>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>

#include <pcl/filters/voxel_grid.h>
#include <pclomp/ndt_omp.h>
#include <fast_gicp/ndt/ndt_cuda.hpp>

#include <hdl_localization/pose_estimator.hpp>
#include <hdl_localization/delta_estimater.hpp>

#include <hdl_global_localization/srv/scan_matching_status.hpp>
#include <hdl_global_localization/srv/set_global_map.hpp>
#include <hdl_global_localization/srv/query_global_localization.hpp>

namespace hdl_localization {

class HdlLocalizationNode : public rclcpp::Node {
    public:
        using PointT = pcl::PointXYZI;
        using std::placeholders;

        HdlLocalizationNode(const std::string node_name, const rclcpp::NodeOptions& options) :: HdlLocalizationNode(node_name, options) : tf_buffer(), tf_listener(tf_buffer){
            // Decide the PointCloud Registration Method and Nearest PointCloud Search Method
            // Then initialize the delta_estimater and the pose_estimater
            initialize_parameters();

            // Get the frame id of the odometry and robot_base
            robot_odom_frame_id = this->declare_parameter<std::string>("robot_odom_frame_id", "odom");
            odom_child_frame_id = this->declare_parameter<std::string>("odom_child_frame_id", "base_link");

            // Use IMU Data or not and invert the acc data and gyro data in IMU Data or not.
            use_imu = this->declare_parameter<bool>("use_imu", true);
            invert_acc = this->declare_parameter<bool>("invert_acc", false);
            invert_gyro = this->declare_parameter<bool>("invert_gyro", false);

            // If use IMU, subscribe the IMU Topic
            if (use_imu) {
            RCLCPP_INFO(get_logger(), "enable imu-based prediction");
            imu_sub = this->create_subscription<sensor_msgs::msg::Imu>("imu_data", 256, std::bind(&HdlLocalizationNode::imu_callback, this, _1));
            }
            // Subscribe Real-Time scanning Point Cloud Topic
            points_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>("velodyne_points", 5, std::bind(&HdlLocalizationNode::points_callback, this, _1));
            // Subscribe global Point Cloud Map which is published by hdl_global_localization_nodlet
            globalmap_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>("globalmap", 1, std::bind(&HdlLocalizationNode::globalmap_callback, this, _1));
            // Subscribe the initial pose
            initialpose_sub = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>("initialpose", 8, std::bind(&HdlLocalizationNode::initialpose_callback, this, _1));

            // Odometry Publisher
            pose_pub = this->create_publisher<nav_msgs::msg::Odometry>("odom", 5);
            // Aligned PointCloud Publisher
            aligned_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("aligned_points", 5);
            // ScanMatchingStatus Publisher
            status_pub = this->create_publisher<hdl_localization::msg::ScanMatchingStatus>("status", 5);
        }

        virtual ~HdlLocalizationNode() {

        }
    
    private:
        pcl::Registration<PointT, PointT>::Ptr create_registration() const {
            // Optional Point Cloud Registration: NDT_OMP, NDT_CUDA_P2D, NDT_CUDA_D2D
            // Name                       Speed         Accuarity       Description
            // NDT_OMP(CPU Only)          Normal        Normal          CPU Distributions to Distributions
            // NDT_CUDA_P2D(GPU Needed)   Normal        High            GPU Points to Distributions
            // NDT_CUDA_D2D(GPU Needed)   Fast          Normal          GPU Distributions to Distributions
            std::string reg_method = this->declare_parameter<std::string>("reg_method", "NDT_OMP");

            // Optional ndt_neighbor_search_method: DIRECT1, DIRECT7, DIRECT_RADIUS, KDTREE
            // Name             Speed         Accuarity       Description
            // DIRECT1          Fast          Low             Search the nearest cell
            // DIRECT7          Normal        Normal          Search 7 nearest cells 
            // DIRECT_RADIUS    Slow          High            Search all cells in the range of radius which is set by users
            // KDTREE           High          High            Available to find the nearest cell fastly and accurately but cost plenty of RAM
            std::string ndt_neighbor_search_method = this->declare_parameter<std::string>("ndt_neighbor_search_method", "DIRECT7");

            // If DIRECT_RADIUS method is selected, set the radius which is the range to search.
            double ndt_neighbor_search_radius = this->declare_parameter<double>("ndt_neighbor_search_radius", 2.0);

            // The scale of voxel
            double ndt_resolution = this->declare_parameter<double>("ndt_resolution", 1.0);

            // Point Cloud Registration Method Option
            if (reg_method == "PCL_OMP")
            {
                RCLCPP_INFO(get_logger(), "PCL_OMP is selected.");
                pclomp::NormalDistributionsTransform<PointT, PointT>::Ptr ndt(new pclomp::NormalDistributionsTransform<PointT, PointT>());
                ndt->setTransformationEpsilon(0.01);
                ndt->setResolution(ndt_resolution);
                // Nearest Cell Method Option
                if (ndt_neighbor_search_method == "DIRECT1") {
                    RCLCPP_INFO(get_logger(), "search_method DIRECT1 is selected");
                    ndt->setNeighborhoodSearchMethod(pclomp::DIRECT1);
                } else if (ndt_neighbor_search_method == "DIRECT7") {
                    RCLCPP_INFO(get_logger(), "search_method DIRECT7 is selected");
                    ndt->setNeighborhoodSearchMethod(pclomp::DIRECT7);
                } else {
                    if (ndt_neighbor_search_method == "KDTREE") {
                    RCLCPP_INFO(get_logger(), "search_method KDTREE is selected");
                    } else {
                    RCLCPP_WARN(get_logger(), "invalid search method was given");
                    RCLCPP_WARN(get_logger(), "default method is selected (KDTREE)");
                    }
                    ndt->setNeighborhoodSearchMethod(pclomp::KDTREE);
                }
                return ndt;
                } else if(reg_method.find("NDT_CUDA") != std::string::npos) {
                RCLCPP_INFO(get_logger(), "NDT_CUDA is selected");
                boost::shared_ptr<fast_gicp::NDTCuda<PointT, PointT>> ndt(new fast_gicp::NDTCuda<PointT, PointT>);
                ndt->setResolution(ndt_resolution);

                if(reg_method.find("D2D") != std::string::npos) {
                    ndt->setDistanceMode(fast_gicp::NDTDistanceMode::D2D);
                } else if (reg_method.find("P2D") != std::string::npos) {
                    ndt->setDistanceMode(fast_gicp::NDTDistanceMode::P2D);
                }

                if (ndt_neighbor_search_method == "DIRECT1") {
                    RCLCPP_INFO(get_logger(), "search_method DIRECT1 is selected");
                    ndt->setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT1);
                } else if (ndt_neighbor_search_method == "DIRECT7") {
                    RCLCPP_INFO(get_logger(), "search_method DIRECT7 is selected");
                    ndt->setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT7);
                } else if (ndt_neighbor_search_method == "DIRECT_RADIUS") {
                    RCLCPP_INFO_STREAM(get_logger(), "search_method DIRECT_RADIUS is selected : " << ndt_neighbor_search_radius);
                    ndt->setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT_RADIUS, ndt_neighbor_search_radius);
                } else {
                    RCLCPP_WARN(get_logger(), "invalid search method was given");
                }
                return ndt;
            }

            RCLCPP_ERROR_STREAM(get_logger(), "unknown registration method:" << reg_method);
            return nullptr;
        }
        
        void initialize_parameters() {
            // intialize scan matching method
            auto downsample_resolution = this->declare_parameter<double>("downsample_resolution", 0.1);
            std::shared_ptr<pcl::VoxelGrid<PointT>> voxelgrid(newpcl::VoxelGrid<PointT>());
            voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
            downsample_filter = voxelgrid;

            RCLCPP_INFO(get_logger(), "create registration method for localization");
            registration = create_registration();

            // global localization
            RCLCPP_INFO(get_logger(), "create registration method for fallback during relocalization");
            relocalizing = false;

            // Reset the memory of the Object delta_estimater which is the instance of DeltaEstimater
            delta_estimater.reset(new DeltaEstimater(create_registration()));

            // initialize pose estimator, if the initial pose is {ZERO}
            if(this->declare_parameter<bool>("specify_init_pose", true)) 
            {
                RCLCPP_INFO(get_logger(), "initialize pose estimator with specified parameters!!");
                pose_estimator.reset(new hdl_localization::PoseEstimator(
                    registration,
                    this->get_clock()->now(),
                    Eigen::Vector3f(this->declare_parameter<double>("init_pos_x", 0.0), this->declare_parameter<double>("init_pos_y", 0.0), this->declare_parameter<double>("init_pos_z", 0.0)),
                    Eigen::Quaternionf(this->declare_parameter<double>("init_ori_w", 1.0), this->declare_parameter<double>("init_ori_x", 0.0), this->declare_parameter<double>("init_ori_y", 0.0), this->declare_parameter<double>("init_ori_z", 0.0)),
                    this->declare_parameter<double>("cool_time_duration", 0.5)
                ));
            
            }
        }

        /**
         * @brief callback for imu data
         * @param imu_msg
         */            
        void imu_callback(sensor_msgs::msg::Imu::ConstSharedPtr imu_msg){
            std::lock_guard<std::mutex> lock(imu_data_mutex);
            imu_data_list.push_back(imu_msg);

        }

        /**
         * @brief callback for point cloud data
         * @param points_msg
         */
        void points_callback(sensor_msgs::msg::PointCloud2::ConstSharedPtr points_msg){
            std::lock_guard<std::mutex> estimator_lock(pose_estimator_mutex);
            if(!pose_estimator) {
            RCLCPP_ERROR(get_logger(), "waiting for initial pose input!!");
            return;
            }

            if(!globalmap) {
            RCLCPP_ERROR(get_logger(), "globalmap has not been received!!");
            return;
            }

            // When Global PointCloud Map and pose_estimator is ok, use Real-Time PointCloud Scanning to estimate the Pose
            const auto& stamp = points_msg->header.stamp;
            pcl::PointCloud<PointT>::Ptr pcl_cloud(new pcl::PointCloud<PointT>());
            pcl::fromROSMsg(*points_msg, *pcl_cloud);

            if(pcl_cloud->empty()) {
            RCLCPP_ERROR(get_logger(), "cloud is empty!!");
            return;
            }

            // transform pointcloud into odom_child_frame_id
            std::string tfError;
            pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
            if(this->tf_buffer.canTransform(odom_child_frame_id, pcl_cloud->header.frame_id, stamp, rclcpp::Duration(0.1), &tfError))
            {
                if(!pcl_ros::transformPointCloud(odom_child_frame_id, *pcl_cloud, *cloud, this->tf_buffer)) {
                    RCLCPP_ERROR(get_logger(), "point cloud cannot be transformed into target frame!!");
                    return;
                }
            }else
            {
                RCLCPP_ERROR(get_logger(), tfError.c_str());
                return;
            }

            auto filtered = downsample(cloud);
            last_scan = filtered;

            if(relocalizing) {
            delta_estimater->add_frame(filtered);
            }

            Eigen::Matrix4f before = pose_estimator->matrix();

            // predict
            if(!use_imu) {
            pose_estimator->predict(stamp);
            } else {
            std::lock_guard<std::mutex> lock(imu_data_mutex);
            auto imu_iter = imu_data_list.begin();
            for(imu_iter; imu_iter != imu_data_list.end(); imu_iter++) {
                if(stamp < (*imu_iter)->header.stamp) {
                break;
                }
                const auto& acc = (*imu_iter)->linear_acceleration;
                const auto& gyro = (*imu_iter)->angular_velocity;
                double acc_sign = invert_acc ? -1.0 : 1.0;
                double gyro_sign = invert_gyro ? -1.0 : 1.0;
                pose_estimator->predict((*imu_iter)->header.stamp, acc_sign * Eigen::Vector3f(acc.x, acc.y, acc.z), gyro_sign * Eigen::Vector3f(gyro.x, gyro.y, gyro.z));
            }
            imu_data_list.erase(imu_data_list.begin(), imu_iter);
            }

            // odometry-based prediction
            rclcpp::Time last_correction_time = pose_estimator->last_correction_time();
            if(private_nh.param<bool>("enable_robot_odometry_prediction", false) && !last_correction_time.isZero()) {
            geometry_msgs::msg::TransformStamped odom_delta;
            if(tf_buffer.canTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, stamp, robot_odom_frame_id, rclcpp::Duration(0, std::pow(10, 8)))) {
                odom_delta = tf_buffer.lookupTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, stamp, robot_odom_frame_id, rclcpp::Duration(0, 0));
            } else if(tf_buffer.canTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, rclcpp::Time(0, 0), robot_odom_frame_id, rclcpp::Duration(0, 0))) {
                odom_delta = tf_buffer.lookupTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, rclcpp::Time(0, 0), robot_odom_frame_id, rclcpp::Duration(0, 0));
            }

            if(odom_delta.header.stamp.nanoseconds() == 0) {
                RCLCPP_WARN_STREAM(get_logger(), "failed to look up transform between " << cloud->header.frame_id << " and " << robot_odom_frame_id);
            } else {
                Eigen::Isometry3d delta = tf2::transformToEigen(odom_delta);
                pose_estimator->predict_odom(delta.cast<float>().matrix());
            }
            }

            // correct
            auto aligned = pose_estimator->correct(stamp, filtered);

            if(aligned_pub->get_subscription_count()) {
            aligned->header.frame_id = "map";
            aligned->header.stamp = cloud->header.stamp;
            aligned_pub->publish(aligned);
            }

            if(status_pub->get_subscription_count()) {
            publish_scan_matching_status(points_msg->header, aligned);
            }

            publish_odometry(points_msg->header.stamp, pose_estimator->matrix());
        }

        /**
         * @brief callback for globalmap input
         * @param points_msg
         */
        void globalmap_callback(sensor_msgs::msg::PointCloud2::ConstSharedPtr points_msg) {
            RCLCPP_INFO(get_logger(), "globalmap received!");
            pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
            pcl::fromROSMsg(*points_msg, *cloud);
            globalmap = cloud;

            registration->setInputTarget(globalmap);

            if(use_global_localization) {
            RCLCPP_INFO(get_logger(), "set globalmap for global localization!");
            hdl_global_localization::SetGlobalMap srv;
            pcl::toROSMsg(*globalmap, srv.request.global_map);

            if(!set_global_map_service.call(srv)) {
                RCLCPP_INFO(get_logger(), "failed to set global map");
            } else {
                RCLCPP_INFO(get_logger(), "done");
            }
            }
        }
        
        /**
         * @brief callback for initial pose input ("2D Pose Estimate" on rviz)
         * @param pose_msg
         */
        void initialpose_callback(geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr pose_msg) {
            RCLCPP_INFO(get_logger(), "initial pose received!!");
            std::lock_guard<std::mutex> lock(pose_estimator_mutex);
            const auto& p = pose_msg->pose.pose.position;
            const auto& q = pose_msg->pose.pose.orientation;
            pose_estimator.reset(
                new hdl_localization::PoseEstimator(
                    registration,
                    this->get_clock()->now(),
                    Eigen::Vector3f(p.x, p.y, p.z),
                    Eigen::Quaternionf(q.w, q.x, q.y, q.z),
                    this->declare_parameter<double>("cool_time_duration", 0.5))
            );
        }

        /**
         * @brief publish scan matching status information
         */
        void publish_scan_matching_status(const std_msgs::msg::Header& header, pcl::PointCloud<pcl::PointXYZI>::ConstPtr aligned) {
            hdl_localization::msg::ScanMatchingStatus status;
            status.header = header;

            status.has_converged = registration->hasConverged();
            status.matching_error = registration->getFitnessScore();

            const double max_correspondence_dist = 0.5;

            int num_inliers = 0;
            std::vector<int> k_indices;
            std::vector<float> k_sq_dists;
            for(int i = 0; i < aligned->size(); i++) {
            const auto& pt = aligned->at(i);
            registration->getSearchMethodTarget()->nearestKSearch(pt, 1, k_indices, k_sq_dists);
            if(k_sq_dists[0] < max_correspondence_dist * max_correspondence_dist) {
                num_inliers++;
            }
            }
            status.inlier_fraction = static_cast<float>(num_inliers) / aligned->size();
            status.relative_pose = tf2::eigenToTransform(Eigen::Isometry3d(registration->getFinalTransformation().cast<double>())).transform;

            status.prediction_labels.reserve(2);
            status.prediction_errors.reserve(2);

            std::vector<double> errors(6, 0.0);

            if(pose_estimator->wo_prediction_error()) {
            status.prediction_labels.push_back(std_msgs::msg::String());
            status.prediction_labels.back().data = "without_pred";
            status.prediction_errors.push_back(tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator->wo_prediction_error().get().cast<double>())).transform);
            }

            if(pose_estimator->imu_prediction_error()) {
            status.prediction_labels.push_back(std_msgs::msg::String());
            status.prediction_labels.back().data = use_imu ? "imu" : "motion_model";
            status.prediction_errors.push_back(tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator->imu_prediction_error().get().cast<double>())).transform);
            }

            if(pose_estimator->odom_prediction_error()) {
            status.prediction_labels.push_back(std_msgs::msg::String());
            status.prediction_labels.back().data = "odom";
            status.prediction_errors.push_back(tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator->odom_prediction_error().get().cast<double>())).transform);
            }

            status_pub->publish(status);
        }

        /**
         * @brief publish odometry
         * @param stamp  timestamp
         * @param pose   odometry pose to be published
         */
        void publish_odometry(const rclcpp::Time& stamp, const Eigen::Matrix4f& pose) {
            // broadcast the transform over tf
            if(tf_buffer.canTransform(robot_odom_frame_id, odom_child_frame_id, rclcpp::Time(0, 0))) {
            geometry_msgs::msg::TransformStamped map_wrt_frame = tf2::eigenToTransform(Eigen::Isometry3d(pose.inverse().cast<double>()));
            map_wrt_frame.header.stamp = stamp;
            map_wrt_frame.header.frame_id = odom_child_frame_id;
            map_wrt_frame.child_frame_id = "map";

            geometry_msgs::msg::TransformStamped frame_wrt_odom = tf_buffer.lookupTransform(robot_odom_frame_id, odom_child_frame_id, rclcpp::Time(0), rclcpp::Duration(std::pow(10, 8)));
            Eigen::Matrix4f frame2odom = tf2::transformToEigen(frame_wrt_odom).cast<float>().matrix();

            geometry_msgs::msg::TransformStamped map_wrt_odom;
            tf2::doTransform(map_wrt_frame, map_wrt_odom, frame_wrt_odom);

            tf2::Transform odom_wrt_map;
            tf2::fromMsg(map_wrt_odom.transform, odom_wrt_map);
            odom_wrt_map = odom_wrt_map.inverse();

            geometry_msgs::msg::TransformStamped odom_trans;
            odom_trans.transform = tf2::toMsg(odom_wrt_map);
            odom_trans.header.stamp = stamp;
            odom_trans.header.frame_id = "map";
            odom_trans.child_frame_id = robot_odom_frame_id;

            tf_broadcaster.sendTransform(odom_trans);
            } else {
            geometry_msgs::msg::TransformStamped odom_trans = tf2::eigenToTransform(Eigen::Isometry3d(pose.cast<double>()));
            odom_trans.header.stamp = stamp;
            odom_trans.header.frame_id = "map";
            odom_trans.child_frame_id = odom_child_frame_id;
            tf_broadcaster.sendTransform(odom_trans);
            }

            // publish the transform
            nav_msgs::msg::Odometry odom;
            odom.header.stamp = stamp;
            odom.header.frame_id = "map";

            tf::poseEigenToMsg(Eigen::Isometry3d(pose.cast<double>()), odom.pose.pose);
            odom.child_frame_id = odom_child_frame_id;
            odom.twist.twist.linear.x = 0.0;
            odom.twist.twist.linear.y = 0.0;
            odom.twist.twist.angular.z = 0.0;

            pose_pub->publish(odom);
        }

        /**
         * @brief downsampling
         * @param cloud   input cloud
         * @return downsampled cloud
         */
        pcl::PointCloud<PointT>::ConstPtr downsample(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
            if(!downsample_filter) {
            return cloud;
            }

            pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
            downsample_filter->setInputCloud(cloud);
            downsample_filter->filter(*filtered);
            filtered->header = cloud->header;

            return filtered;
        }

    private:
        std::string robot_odom_frame_id;
        std::string odom_child_frame_id;

        tf2_ros::Buffer tf_buffer;
        tf2_ros::TransformListener tf_listener;
        tf2_ros::TransformBroadcaster tf_broadcaster;

        bool use_imu;
        bool invert_acc;
        bool invert_gyro;

        // imu input buffer
        std::mutex imu_data_mutex;
        std::vector<sensor_msgs::msg::Imu::ConstPtr> imu_data_list;

        // globalmap and registration method
        pcl::PointCloud<PointT>::Ptr globalmap;
        pcl::Filter<PointT>::Ptr downsample_filter;
        pcl::Registration<PointT, PointT>::Ptr registration;

        // pose estimator
        std::mutex pose_estimator_mutex;
        std::unique_ptr<hdl_localization::PoseEstimator> pose_estimator;

        // Subscription
        rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub;
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr points_sub;
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr globalmap_sub;
        rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initialpose_sub;

        // Publisher
        rclcpp::Publisher<nav_msgs::Odometry>::SharedPtr pose_pub;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr aligned_pub;
        rclcpp::Publisher<hdl_localization::msg::ScanMatchingStatus>::SharedPtr status_pub;

};

}

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    options.automatically_declare_parameters_from_overrides(true);
    rclcpp::executors::MultiThreadedExecutor excutor;
    auto node = std::make_shared<hdl_localization::HdlLocalizationNode>("hdl_localization_node", options);

    excutor.add_node(node);
    excutor.spin();

    rclcpp::shutdown();
    return 0;
}

