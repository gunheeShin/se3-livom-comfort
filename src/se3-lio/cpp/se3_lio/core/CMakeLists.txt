
add_library(core SHARED)
add_library(se3_lio::core ALIAS core)
target_sources(core PRIVATE state_predict.cpp lie.cpp state_update.cpp map_management.cpp temporary/voxel_map_util.cpp)
target_include_directories(core PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(core Eigen3::Eigen Sophus::Sophus)