add_library(visual SHARED)
add_library(se3_lio::visual ALIAS visual)
target_sources(visual PRIVATE camera.cpp visual_map.cpp visual_update.cpp)
target_include_directories(visual PUBLIC ${PROJECT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(visual se3_lio::core ${OpenCV_LIBS})
target_compile_features(visual PUBLIC cxx_std_17)

add_executable(test_visual tests/test_visual.cpp)
target_link_libraries(test_visual se3_lio::visual)
