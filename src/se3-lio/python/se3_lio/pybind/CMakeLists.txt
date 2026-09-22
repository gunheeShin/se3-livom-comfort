pybind11_add_module(se3_lio_pybind MODULE se3_lio_pybind.cpp)
target_link_libraries(
  se3_lio_pybind
  PRIVATE se3_lio::pipeline
          se3_lio::core
          se3_lio::synchronizer
          se3_lio::backend)

# The core is built as SHARED libraries that the core's own CMakeLists installs
# under <install-dir>/lib/se3_lio. The module sits at <install-dir>/, so point
# its RPATH there; the core libs depend on each other in the same dir ($ORIGIN).
set_target_properties(se3_lio_pybind PROPERTIES
  INSTALL_RPATH "$ORIGIN/lib/se3_lio")
set_target_properties(pipeline core synchronizer visual backend PROPERTIES
  INSTALL_RPATH "$ORIGIN")

install(TARGETS se3_lio_pybind DESTINATION .)
