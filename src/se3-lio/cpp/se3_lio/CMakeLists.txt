cmake_minimum_required(VERSION 3.16...3.26)
project(se3_lio_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenMP QUIET REQUIRED)

if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

option(USE_CCACHE "Build using Ccache if found on the path" ON)
option(USE_SYSTEM_EIGEN3 "Use system pre-installed Eigen" ON)
option(USE_SYSTEM_SOPHUS "Use system pre-installed Sophus" ON)
include(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/find_dependencies.cmake)

# Setup build options

find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d)

add_subdirectory(synchronizer)
add_subdirectory(core)
add_subdirectory(visual)
add_subdirectory(pipeline)
add_subdirectory(backend)

install(TARGETS synchronizer DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/se3_lio)
install(TARGETS pipeline DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/se3_lio)
install(TARGETS core DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/se3_lio)
install(TARGETS visual DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/se3_lio)
install(TARGETS backend DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/se3_lio)


install(TARGETS synchronizer pipeline core visual backend
    EXPORT se3_lioTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/se3_lio
    INCLUDES DESTINATION ${CMAKE_INSTALL_PREFIX}/include/se3_lio
)

install(DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/synchronizer
    ${CMAKE_CURRENT_SOURCE_DIR}/visual
    ${CMAKE_CURRENT_SOURCE_DIR}/backend
    DESTINATION ${CMAKE_INSTALL_PREFIX}/include/se3_lio
    FILES_MATCHING PATTERN "*.h"
)