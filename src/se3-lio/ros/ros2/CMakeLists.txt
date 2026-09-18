cmake_minimum_required(VERSION 3.16...3.26)

unset(CATKIN_INSTALL_INTO_PREFIX_ROOT)
unset(CATKIN_SYMLINK_INSTALL)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(std_msgs REQUIRED)

find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(livox_ros_driver2 REQUIRED)

# LiDAR input type — selected at build time, no source edits:
#   colcon build --packages-select se3_lio --cmake-args -DLIDAR_TYPE=ouster
set(LIDAR_TYPE "livox" CACHE STRING "LiDAR input type: ouster | livox")
if(LIDAR_TYPE STREQUAL "livox")
    add_compile_definitions(LIDAR_LIVOX)
elseif(LIDAR_TYPE STREQUAL "ouster")
    add_compile_definitions(LIDAR_OUSTER)
else()
    message(FATAL_ERROR "Unknown LIDAR_TYPE '${LIDAR_TYPE}' (expected: ouster | livox)")
endif()
message(STATUS "SE3-LIO LiDAR input type: ${LIDAR_TYPE}")

set(SE3_LIO_CPP_INSTALL_DIR ${PROJECT_SOURCE_DIR}/../cpp/se3_lio/install)
set(SE3_LIO_CPP_DIR ${PROJECT_SOURCE_DIR}/../cpp/se3_lio)
if(EXISTS ${SE3_LIO_CPP_INSTALL_DIR})
    message(STATUS "Using installed SE3-LIO C++ core library from ${SE3_LIO_CPP_INSTALL_DIR}...")
    link_directories(${SE3_LIO_CPP_INSTALL_DIR}/lib/se3_lio)
    add_executable(lio_node src/lio_node.cpp)
    ament_target_dependencies(lio_node
        rclcpp
        nav_msgs
        sensor_msgs
        geometry_msgs
        pcl_conversions
        livox_ros_driver2
        tf2_ros
    )
    target_link_libraries(lio_node
        core synchronizer pipeline Sophus::Sophus
    )
    target_include_directories(lio_node
        PRIVATE
        ${SE3_LIO_CPP_INSTALL_DIR}/include/se3_lio
    )
    install(DIRECTORY
        launch
        config
        rviz
        DESTINATION share/${PROJECT_NAME}
    )

    install(TARGETS lio_node
        DESTINATION lib/${PROJECT_NAME}
    )
elseif (EXISTS ${SE3_LIO_CPP_DIR})
    message(STATUS "Building SE3-LIO C++ core library from source...")
    # The core libs (core/synchronizer/pipeline) install side-by-side with
    # lio_node into lib/se3_lio. RPATH=$ORIGIN lets `ros2 run` resolve them at
    # runtime — a from-source colcon install otherwise strips the build RPATH,
    # so the executable can't find libsynchronizer.so/libcore.so/libpipeline.so.
    # Set before add_subdirectory so the core targets inherit it too.
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    add_subdirectory(${PROJECT_SOURCE_DIR}/../cpp/se3_lio ${CMAKE_CURRENT_BINARY_DIR}/se3_lio_cpp)
    include_directories(${PROJECT_SOURCE_DIR}/../cpp/se3_lio)

    add_executable(lio_node src/lio_node.cpp)

    target_link_libraries(lio_node
        se3_lio::synchronizer se3_lio::pipeline
    )
    ament_target_dependencies(lio_node
        rclcpp
        nav_msgs
        sensor_msgs
        geometry_msgs
        pcl_conversions
        livox_ros_driver2
        tf2_ros
    )
    target_include_directories(lio_node
        PRIVATE
        ${PCL_INCLUDE_DIRS}
    )

    install(DIRECTORY
        launch
        config
        rviz
        DESTINATION share/${PROJECT_NAME}
    )

    install(TARGETS lio_node
        DESTINATION lib/${PROJECT_NAME}
    )
else()
    message(FATAL_ERROR "SE3-LIO C++ core library not found. Please build the C++ core first.")
endif()

# Python (rclpy) node — thin wrapper over the pybind `se3_lio` core.
# Run with: ros2 run se3_lio se3_lio_py_node  /  ros2 launch se3_lio run_py.launch.py
install(PROGRAMS
    scripts/se3_lio_node.py
    DESTINATION lib/${PROJECT_NAME}
    RENAME se3_lio_py_node
)

ament_package()

