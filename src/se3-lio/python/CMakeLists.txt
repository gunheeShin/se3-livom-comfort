cmake_minimum_required(VERSION 3.16...3.26)
project(se3_lio_pybind VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(PYBIND11_NEWPYTHON ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/se3_lio/)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../cpp/se3_lio
                   ${CMAKE_CURRENT_BINARY_DIR}/se3_lio_cpp)
else()
  cmake_minimum_required(VERSION 3.18)
  message(STATUS "Performing out-of-tree build, fetching SE3-LIO v${CMAKE_PROJECT_VERSION} Release from Github")
  include(FetchContent)
  FetchContent_Declare(
    ext_se3_lio_core PREFIX se3_lio_core
    URL https://github.com/url-kaist/se3-lio/archive/refs/tags/v${CMAKE_PROJECT_VERSION}.tar.gz
    SOURCE_SUBDIR cpp/se3_lio)
  FetchContent_MakeAvailable(ext_se3_lio_core)
endif()

add_subdirectory(se3_lio/pybind)
