#include <mutex>
#include <memory>
#include <iostream>

#include <rclcpp/rclcpp.hpp>
#include <pcl_ros/transforms.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>

#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <std_srvs/srv/empty.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>

#include <pcl/filters/voxel_grid.h>
#include <pclomp/ndt_omp.h>
#include <fast_gicp/ndt/ndt_cuda.hpp>
#include <small_gicp/pcl/pcl_registration.hpp>

#include <hdl_localization/pose_estimator.hpp>
#include <hdl_localization/delta_estimater.hpp>

#include <hdl_localization/msg/scan_matching_status.hpp>
#include <hdl_global_localization/srv/set_global_map.hpp>
#include <hdl_global_localization/srv/query_global_localization.hpp>

namespace hdl_localization {

class HdlLocalizationNode : public rclcpp::Node {
    public:
        using PointT = pcl::PointXYZI;

        HdlLocalizationNode(const std::string node_name, const rclcpp::NodeOptions& options) : Node(node_name, options){

            tf_buffer = std::make_unique<tf2_ros::Buffer>(get_clock());
            tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);
            tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(this);

            robot_odom_frame_id = declare_parameter<std::string>("robot_odom_frame_id", "odom");
            odom_child_frame_id = declare_parameter<std::string>("odom_child_frame_id", "base_link");
            send_tf_transforms = declare_parameter<bool>("send_tf_transforms", true);
            cool_time_duration = declare_parameter<double>("cool_time_duration", 0.5);
            // Optional Point Cloud Registration: NDT_OMP, NDT_CUDA_P2D, NDT_CUDA_D2D
            // Name                       Speed         Accuarity       Description
            // NDT_OMP(CPU Only)          Normal        Normal          CPU Distributions to Distributions
            // NDT_CUDA_P2D(GPU Needed)   Normal        High            GPU Points to Distributions
            // NDT_CUDA_D2D(GPU Needed)   Fast          Normal          GPU Distributions to Distributions            
            reg_method = declare_parameter<std::string>("reg_method", "NDT_OMP");
            // Optional ndt_neighbor_search_method: DIRECT1, DIRECT7, DIRECT_RADIUS, KDTREE
            // Name             Speed         Accuarity       Description
            // DIRECT1          Fast          Low             Search the nearest cell
            // DIRECT7          Normal        Normal          Search 7 nearest cells 
            // DIRECT_RADIUS    Slow          High            Search all cells in the range of radius which is set by users
            // KDTREE           High          High            Available to find the nearest cell fastly and accurately but cost plenty of RAM            
            ndt_neighbor_search_method = declare_parameter<std::string>("ndt_neighbor_search_method", "DIRECT7");
            // If DIRECT_RADIUS method is selected, set the radius which is the range to search.
            ndt_neighbor_search_radius = declare_parameter<double>("ndt_neighbor_search_radius", 2.0);
            // The scale of voxel
            ndt_resolution = declare_parameter<double>("ndt_resolution", 1.0);
            enable_robot_odometry_prediction = declare_parameter<bool>("enable_robot_odometry_prediction", false);
            gicp_thread_num = declare_parameter<int>("gicp_thread_num", 4);
            gicp_neighbors_num = declare_parameter<int>("gicp_neighbors_num", 20);
            gicp_correspondence_distance = declare_parameter<double>("gicp_correspondence_distance", 1.0);
            gicp_voxel_resolution = declare_parameter<double>("gicp_voxel_resolution", 1.0);
            // Get the frame id of the odometry and robot_base
            // Use IMU Data or not and invert the acc data and gyro data in IMU Data or not.
            use_imu = this->declare_parameter<bool>("use_imu", true);
            invert_acc = this->declare_parameter<bool>("invert_acc", false);
            invert_gyro = this->declare_parameter<bool>("invert_gyro", false);

            // If use IMU, subscribe the IMU Topic
            if (use_imu) {
                RCLCPP_INFO(get_logger(), "enable imu-based prediction");
                imu_sub = this->create_subscription<sensor_msgs::msg::Imu>("imu_data", 256, std::bind(&HdlLocalizationNode::imu_callback, this, std::placeholders::_1));
            }

            auto latch_qos = rclcpp::QoS(1).transient_local();
            // Subscribe Real-Time scanning Point Cloud Topic
            points_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>("velodyne_points", 5, std::bind(&HdlLocalizationNode::points_callback, this, std::placeholders::_1));
            // Subscribe global Point Cloud Map which is published by hdl_global_localization_nodlet
            globalmap_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>("globalmap", latch_qos, std::bind(&HdlLocalizationNode::globalmap_callback, this, std::placeholders::_1));
            // Subscribe the initial pose
            initialpose_sub = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>("initialpose", 8, std::bind(&HdlLocalizationNode::initialpose_callback, this, std::placeholders::_1));

            // Odometry Publisher
            pose_pub = this->create_publisher<nav_msgs::msg::Odometry>("odom", 5);
            // Aligned PointCloud Publisher
            aligned_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("aligned_points", 5);
            // ScanMatchingStatus Publisher
            status_pub = this->create_publisher<hdl_localization::msg::ScanMatchingStatus>("status", 5);

            // global localization
            use_global_localization = declare_parameter<bool>("use_global_localization", true);
            if (use_global_localization) {
                RCLCPP_INFO_STREAM(get_logger(), "wait for global localization services");
                set_global_map_service = create_client<hdl_global_localization::srv::SetGlobalMap>("set_global_map");
                query_global_localization_service = create_client<hdl_global_localization::srv::QueryGlobalLocalization>("query");
                while (!set_global_map_service->wait_for_service(std::chrono::milliseconds(1000))) {
                    RCLCPP_WARN(get_logger(), "Waiting for SetGlobalMap service");
                    if (!rclcpp::ok()) {
                        return;
                    }
                }
                while (!query_global_localization_service->wait_for_service(std::chrono::milliseconds(1000))) {
                    RCLCPP_WARN(get_logger(), "Waiting for QueryGlobalLocalization service");
                    if (!rclcpp::ok()) {
                        return;
                    }
                }

                relocalize_server = create_service<std_srvs::srv::Empty>("relocalize", std::bind(&HdlLocalizationNode::relocalize, this, std::placeholders::_1, std::placeholders::_2));
            }

            // Decide the PointCloud Registration Method and Nearest PointCloud Search Method
            // Then initialize the delta_estimater and the pose_estimater
            initialize_parameters();
        }

        virtual ~HdlLocalizationNode() {

        }
    
    private:
        pcl::Registration<PointT, PointT>::Ptr create_registration() {
            if(reg_method == "NDT_OMP") {
                RCLCPP_INFO(get_logger(), "NDT_OMP is selected");
                pclomp::NormalDistributionsTransform<PointT, PointT>::Ptr ndt(new pclomp::NormalDistributionsTransform<PointT, PointT>());
                ndt->setTransformationEpsilon(0.01);
                ndt->setResolution(ndt_resolution);
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
            } else if (reg_method.find("GICP") != std::string::npos) {
                if (reg_method.find("VGICP")) {
                    RCLCPP_INFO(get_logger(), "VGICP is selected");
                    small_gicp::RegistrationPCL<PointT, PointT>::Ptr gicp(new small_gicp::RegistrationPCL<PointT, PointT>());
                    gicp->setNumThreads(gicp_thread_num);
                    gicp->setCorrespondenceRandomness(gicp_neighbors_num);
                    gicp->setMaxCorrespondenceDistance(gicp_correspondence_distance);
                    gicp->setVoxelResolution(gicp_voxel_resolution);
                    gicp->setRegistrationType("VGICP");
                    
                    return gicp;
                } else {
                    RCLCPP_INFO(get_logger(), "GICP is selected");
                    small_gicp::RegistrationPCL<PointT, PointT>::Ptr gicp(new small_gicp::RegistrationPCL<PointT, PointT>());
                    gicp->setNumThreads(gicp_thread_num);
                    gicp->setCorrespondenceRandomness(gicp_neighbors_num);
                    gicp->setMaxCorrespondenceDistance(gicp_correspondence_distance);
                    gicp->setVoxelResolution(gicp_voxel_resolution);
                    gicp->setRegistrationType("GICP");

                    return gicp;
                }
            } else if(reg_method.find("NDT_CUDA") != std::string::npos) {
                RCLCPP_INFO(get_logger(), "NDT_CUDA is selected");
                std::shared_ptr<fast_gicp::NDTCuda<PointT, PointT>> ndt(new fast_gicp::NDTCuda<PointT, PointT>);
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
            std::shared_ptr<pcl::VoxelGrid<PointT>> voxelgrid(new pcl::VoxelGrid<PointT>());
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
                    get_clock()->now(),
                    Eigen::Vector3f(this->declare_parameter<double>("init_pos_x", 0.0), this->declare_parameter<double>("init_pos_y", 0.0), this->declare_parameter<double>("init_pos_z", 0.0)),
                    Eigen::Quaternionf(this->declare_parameter<double>("init_ori_w", 1.0), this->declare_parameter<double>("init_ori_x", 0.0), this->declare_parameter<double>("init_ori_y", 0.0), this->declare_parameter<double>("init_ori_z", 0.0)),
                    cool_time_duration
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
            const auto& stamp = rclcpp::Time(points_msg->header.stamp, get_clock()->get_clock_type());

            pcl::PointCloud<PointT>::Ptr pcl_cloud(new pcl::PointCloud<PointT>());
            pcl::fromROSMsg(*points_msg, *pcl_cloud);

            if(pcl_cloud->empty()) {
                RCLCPP_ERROR(get_logger(), "cloud is empty!!");
                return;
            }

            // transform pointcloud into odom_child_frame_id
            std::string tfError;
            pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
            if(this->tf_buffer->canTransform(odom_child_frame_id, pcl_cloud->header.frame_id, stamp, rclcpp::Duration(0, std::pow(10, 8)), &tfError))
            {
                if(!pcl_ros::transformPointCloud(odom_child_frame_id, *pcl_cloud, *cloud, *this->tf_buffer)) {
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
                    auto imu_stamp = rclcpp::Time((*imu_iter)->header.stamp, get_clock()->get_clock_type());

                    if(stamp < imu_stamp) {
                        break;
                    }
                    const auto& acc = (*imu_iter)->linear_acceleration;
                    const auto& gyro = (*imu_iter)->angular_velocity;
                    double acc_sign = invert_acc ? -1.0 : 1.0;
                    double gyro_sign = invert_gyro ? -1.0 : 1.0;
                    pose_estimator->predict(rclcpp::Time((*imu_iter)->header.stamp, get_clock()->get_clock_type()), acc_sign * Eigen::Vector3f(acc.x, acc.y, acc.z), gyro_sign * Eigen::Vector3f(gyro.x, gyro.y, gyro.z));
                }
                imu_data_list.erase(imu_data_list.begin(), imu_iter);
            }

            // odometry-based prediction
            rclcpp::Time last_correction_time = rclcpp::Time(pose_estimator->last_correction_time(), get_clock()->get_clock_type());
            
            if (enable_robot_odometry_prediction && last_correction_time != rclcpp::Time((int64_t)0, get_clock()->get_clock_type())) {
                geometry_msgs::msg::TransformStamped odom_delta;
                if (tf_buffer->canTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, stamp, robot_odom_frame_id, rclcpp::Duration(std::chrono::milliseconds(100)))) {
                    odom_delta = tf_buffer->lookupTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, stamp, robot_odom_frame_id, rclcpp::Duration(std::chrono::milliseconds(0)));
                } else if(tf_buffer->canTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, rclcpp::Time((int64_t)0, get_clock()->get_clock_type()), robot_odom_frame_id, rclcpp::Duration(std::chrono::milliseconds(0)))) {
                    odom_delta = tf_buffer->lookupTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, rclcpp::Time((int64_t)0, get_clock()->get_clock_type()), robot_odom_frame_id, rclcpp::Duration(std::chrono::milliseconds(0)));
                }

                if(rclcpp::Time(odom_delta.header.stamp, get_clock()->get_clock_type()) == rclcpp::Time((int64_t)0, get_clock()->get_clock_type())) {
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

            sensor_msgs::msg::PointCloud2 globalmap_msg;
            pcl::toROSMsg(*aligned, globalmap_msg);            
            
            aligned_pub->publish(globalmap_msg);
            }

            if(status_pub->get_subscription_count()) {
            publish_scan_matching_status(points_msg->header, aligned);
            }

            publish_odometry(rclcpp::Time(points_msg->header.stamp, get_clock()->get_clock_type()), pose_estimator->matrix());
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
                auto req  = std::make_shared<hdl_global_localization::srv::SetGlobalMap::Request>();                
                pcl::toROSMsg(*globalmap, req->global_map);

                set_global_map_service->wait_for_service();
                set_global_map_service->async_send_request(req, std::bind(&HdlLocalizationNode::set_global_map_callback, this, std::placeholders::_1));
            }
        }

        /**
         * @brief perform global localization to relocalize the sensor position
         * @param
         */
        bool relocalize(std::shared_ptr<std_srvs::srv::Empty::Request> req, std::shared_ptr<std_srvs::srv::Empty::Response> res) {
            if(last_scan == nullptr) {
                RCLCPP_INFO_STREAM(get_logger(), "no scan has been received");
                return false;
            }

            relocalizing = true;
            delta_estimater->reset();
            pcl::PointCloud<PointT>::ConstPtr scan = last_scan;

            auto query_req  = std::make_shared<hdl_global_localization::srv::QueryGlobalLocalization::Request>();
            pcl::toROSMsg(*scan, query_req->cloud);
            query_req->max_num_candidates = 1;

            int i = 0;
            while (!query_global_localization_service->wait_for_service(std::chrono::seconds(1))) {
                RCLCPP_INFO(get_logger(), "Waitting for query global localization service...");
                if (i >= 2) {
                    RCLCPP_ERROR(get_logger(), "Failed to call QueryGlobalLocalization service");
                    return false;
                }
                i = i + 1;
            }

            query_global_localization_service->async_send_request(query_req, std::bind(&HdlLocalizationNode::query_global_localization_callback, this, std::placeholders::_1));

            return true;
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
                    get_clock()->now(),
                    Eigen::Vector3f(p.x, p.y, p.z),
                    Eigen::Quaternionf(q.w, q.x, q.y, q.z),
                    cool_time_duration
                )
            );
        }

        /**
         * @brief callback for get the response from "SetGlobalMap" server
         * @param res_future
         */
        void set_global_map_callback(const rclcpp::Client<hdl_global_localization::srv::SetGlobalMap>::SharedFuture res_future) {
            RCLCPP_INFO(get_logger(), "Done!");
        }

        /**
         * @brief callback for get the response from "QueryGlobalLocalization" server
         * @param res_future
         */
        void query_global_localization_callback(const rclcpp::Client<hdl_global_localization::srv::QueryGlobalLocalization>::SharedFuture res_future) {
            
            auto query_result = res_future.get();
            const auto& result = query_result->poses[0];

            RCLCPP_INFO_STREAM(get_logger(), "--- Global localization result ---");
            RCLCPP_INFO_STREAM(get_logger(), "Trans :" << result.position.x << " " << result.position.y << " " << result.position.z);
            RCLCPP_INFO_STREAM(get_logger(), "Quat  :" << result.orientation.x << " " << result.orientation.y << " " << result.orientation.z << " " << result.orientation.w);
            RCLCPP_INFO_STREAM(get_logger(), "Error :" << query_result->errors[0]);
            RCLCPP_INFO_STREAM(get_logger(), "Inlier:" << query_result->inlier_fractions[0]);

            Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
            pose.linear() = Eigen::Quaternionf(result.orientation.w, result.orientation.x, result.orientation.y, result.orientation.z).toRotationMatrix();
            pose.translation() = Eigen::Vector3f(result.position.x, result.position.y, result.position.z);
            pose = pose * delta_estimater->estimated_delta();

            std::lock_guard<std::mutex> lock(pose_estimator_mutex);
            pose_estimator.reset(new hdl_localization::PoseEstimator(
                registration,
                get_clock()->now(),
                pose.translation(),
                Eigen::Quaternionf(pose.linear()),
                cool_time_duration
            ));

            relocalizing = false;

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
            int stamp_sec = stamp.seconds();
            uint32_t stamp_nanosec = stamp.nanoseconds() % int(1e9);

            if (send_tf_transforms){
                if(tf_buffer->canTransform(robot_odom_frame_id, odom_child_frame_id, rclcpp::Time(0, 0, get_clock()->get_clock_type()))) {
                    geometry_msgs::msg::TransformStamped map_wrt_frame = tf2::eigenToTransform(Eigen::Isometry3d(pose.inverse().cast<double>()));
                    // map_wrt_frame.header.stamp = stamp;
                    map_wrt_frame.header.stamp.sec = stamp_sec;
                    map_wrt_frame.header.stamp.nanosec = stamp_nanosec;
                    map_wrt_frame.header.frame_id = odom_child_frame_id;
                    map_wrt_frame.child_frame_id = "map";

                    geometry_msgs::msg::TransformStamped frame_wrt_odom = tf_buffer->lookupTransform(robot_odom_frame_id, odom_child_frame_id, rclcpp::Time(0, 0, get_clock()->get_clock_type()), rclcpp::Duration(0, std::pow(10, 8)));
                    Eigen::Matrix4f frame2odom = tf2::transformToEigen(frame_wrt_odom).cast<float>().matrix();

                    geometry_msgs::msg::TransformStamped map_wrt_odom;
                    tf2::doTransform(map_wrt_frame, map_wrt_odom, frame_wrt_odom);

                    tf2::Transform odom_wrt_map;
                    tf2::fromMsg(map_wrt_odom.transform, odom_wrt_map);
                    odom_wrt_map = odom_wrt_map.inverse();

                    geometry_msgs::msg::TransformStamped odom_trans;
                    odom_trans.transform = tf2::toMsg(odom_wrt_map);
                    // odom_trans.header.stamp = stamp;
                    odom_trans.header.stamp.sec = stamp_sec;
                    odom_trans.header.stamp.nanosec = stamp_nanosec;
                    odom_trans.header.frame_id = "map";
                    odom_trans.child_frame_id = robot_odom_frame_id;

                    tf_broadcaster->sendTransform(odom_trans);
                } else {
                    geometry_msgs::msg::TransformStamped odom_trans = tf2::eigenToTransform(Eigen::Isometry3d(pose.cast<double>()));
                    // odom_trans.header.stamp = stamp;
                    odom_trans.header.stamp.sec = stamp_sec;
                    odom_trans.header.stamp.nanosec = stamp_nanosec;
                    odom_trans.header.frame_id = "map";
                    odom_trans.child_frame_id = odom_child_frame_id;
                    tf_broadcaster->sendTransform(odom_trans);
                }
            }

            // publish the transform
            nav_msgs::msg::Odometry odom;
            // odom.header.stamp = stamp;
            odom.header.stamp.sec = stamp_sec;
            odom.header.stamp.nanosec = stamp_nanosec;
            odom.header.frame_id = "map";

            odom.pose.pose = tf2::toMsg(Eigen::Isometry3d(pose.cast<double>()));
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
        std::unique_ptr<tf2_ros::Buffer> tf_buffer;
        std::shared_ptr<tf2_ros::TransformListener> tf_listener;
        std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

        bool use_imu;
        bool invert_acc;
        bool invert_gyro;
        bool send_tf_transforms;
        bool enable_robot_odometry_prediction;
        std::string robot_odom_frame_id;
        std::string odom_child_frame_id;
        std::string reg_method;
        std::string ndt_neighbor_search_method;
        double cool_time_duration;
        double ndt_neighbor_search_radius;
        double ndt_resolution;
        double gicp_correspondence_distance;
        double gicp_voxel_resolution;
        int gicp_thread_num;
        int gicp_neighbors_num;

        
        // imu input buffer
        std::mutex imu_data_mutex;
        std::vector<sensor_msgs::msg::Imu::ConstSharedPtr> imu_data_list;

        // globalmap and registration method
        pcl::PointCloud<PointT>::Ptr globalmap;
        pcl::Filter<PointT>::Ptr downsample_filter;
        pcl::Registration<PointT, PointT>::Ptr registration;
        pcl::PointCloud<PointT>::ConstPtr last_scan;

        // global localization
        bool use_global_localization;
        std::atomic_bool relocalizing;
        std::unique_ptr<DeltaEstimater> delta_estimater;

        // pose estimator
        std::mutex pose_estimator_mutex;
        std::unique_ptr<hdl_localization::PoseEstimator> pose_estimator;

        // Subscription
        rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub;
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr points_sub;
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr globalmap_sub;
        rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initialpose_sub;

        // Publisher
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pose_pub;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr aligned_pub;
        rclcpp::Publisher<hdl_localization::msg::ScanMatchingStatus>::SharedPtr status_pub;

        // Server
        rclcpp::Service<std_srvs::srv::Empty>::SharedPtr relocalize_server;

        // Client
        rclcpp::Client<hdl_global_localization::srv::QueryGlobalLocalization>::SharedPtr query_global_localization_service;
        rclcpp::Client<hdl_global_localization::srv::SetGlobalMap>::SharedPtr set_global_map_service;

};

}

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    options.automatically_declare_parameters_from_overrides(false);

    rclcpp::spin(std::make_shared<hdl_localization::HdlLocalizationNode>("hdl_localization_node", options));

    rclcpp::shutdown();
    return 0;
}

