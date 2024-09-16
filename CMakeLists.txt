cmake_minimum_required(VERSION 3.15)
project(hdl_localization)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_auto REQUIRED)
find_package(rosidl_default_generators REQUIRED)

find_package(rclcpp REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(std_msgs REQUIRED)

find_package(ndt_omp REQUIRED)
find_package(fast_gicp REQUIRED)
find_package(small_gicp REQUIRED)
find_package(hdl_global_localization REQUIRED)
find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

message(STATUS "PCL_INCLUDE_DIRS:" ${PCL_INCLUDE_DIRS})
message(STATUS "PCL_LIBRARY_DIRS:" ${PCL_LIBRARY_DIRS})
message(STATUS "PCL_DEFINITIONS:" ${PCL_DEFINITIONS})

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

###################################
### generate message interfaces ###
###################################
find_package(rosidl_default_generators REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ScanMatchingStatus.msg"
  DEPENDENCIES 
  std_msgs 
  geometry_msgs 
)
ament_export_dependencies(rosidl_default_runtime)

###########
## Build ##
###########
# ament_auto_find_build_dependencies()
add_executable(hdl_localization_node 
  src/hdl_localization_node.cpp
  src/hdl_localization/pose_estimator.cpp
)
ament_target_dependencies(hdl_localization_node
  rclcpp
  tf2_ros
  tf2_geometry_msgs
  tf2_eigen
  pcl_ros
  pcl_conversions
  sensor_msgs
  geometry_msgs
  nav_msgs
  std_srvs
  ndt_omp
  fast_gicp
  small_gicp
  hdl_global_localization
)
rosidl_target_interfaces(hdl_localization_node
  ${PROJECT_NAME}
  rosidl_typesupport_cpp
)
target_include_directories(hdl_localization_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)


add_executable(globalmap_server_node 
  src/globalmap_server_node.cpp
)
ament_target_dependencies(globalmap_server_node
  rclcpp
  tf2_ros
  pcl_conversions
  sensor_msgs
  std_msgs
)
target_include_directories(globalmap_server_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# install exec,launch, config, rviz etc.
install(TARGETS hdl_localization_node DESTINATION lib/${PROJECT_NAME})
install(TARGETS globalmap_server_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(DIRECTORY scripts DESTINATION share/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
# install(DIRECTORY rviz DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package()
