# This file is a slightly modified version of:
# https://gitlab.com/tum-vision/lie-spline-experiments/-/blob/master/CMakeLists.txt

cmake_minimum_required(VERSION 3.10)
project(gvi-fusion)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/" ${CMAKE_MODULE_PATH})

# Set default build type if not specified otherwise.
# See https://cmake.org/pipermail/cmake/2012-May/050243.html
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build." FORCE)
  message(STATUS "Setting build type to '${CMAKE_BUILD_TYPE}' as none was specified.")
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

# Configure CCache if available
if (NOT CMAKE_C_COMPILER_LAUNCHER AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    message(STATUS "Found ccache: ${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER   ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
  else()
    message(STATUS "Dind't find ccache")
  endif()
else()
  message(STATUS "Compiler launcher already set. Not configuring ccache.")
  message(STATUS "CMAKE_C_COMPILER_LAUNCHER: ${CMAKE_C_COMPILER_LAUNCHER}")
  message(STATUS "CMAKE_CXX_COMPILER_LAUNCHER: ${CMAKE_CXX_COMPILER_LAUNCHER}")
endif()

SET(CXX_MARCH native)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# for some reason clang doesn't include them in Wall whereas gcc does, so include them explicitly.
SET(EXTRA_WARNING_FLAGS "-Wsign-compare")

SET(CMAKE_CXX_FLAGS " -ftemplate-backtrace-limit=0 -Wall ${EXTRA_WARNING_FLAGS} -march=${CXX_MARCH} ${CMAKE_CXX_FLAGS}")

set(USE_OPENMP_DEFAULT OFF)

# assume libstdc++
set(STD_CXX_FS stdc++fs)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(LINUX_CLANG 1)
  message(STATUS "Detected Linux with clang.")

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  message(STATUS "Detected Linux with gcc.")

else()
  message(WARNING "Detected Linux with unsupported compiler ${CMAKE_CXX_COMPILER_ID}")
endif()

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
#find_package(Eigen3 3.3.7 EXACT REQUIRED MODULE)
set(EIGEN_INCLUDE_DIR ${EIGEN3_INCLUDE_DIR})  # for ceres / opengv
include_directories(${EIGEN3_INCLUDE_DIR})
message(STATUS "Found Eigen headers in: ${EIGEN3_INCLUDE_DIR}")

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

find_package(OpenCV REQUIRED core imgproc calib3d highgui)
include_directories(${OpenCV_INCLUDE_DIR})
message(STATUS "Found OpenCV headers in: ${OpenCV_INCLUDE_DIR}")
message(STATUS "Found OpenCV_LIBS: ${OpenCV_LIBS}")

find_package(yaml-cpp)

include_directories(thirdparty/Sophus)
include_directories(include)

add_executable(fit_spline_to_colmap src/fit_spline_to_colmap.cpp)
target_link_libraries(fit_spline_to_colmap ${CERES_LIBRARIES} yaml-cpp)

add_executable(fit_spline_to_gp_measurements src/fit_spline_to_gp_measurements.cpp)
target_link_libraries(fit_spline_to_gp_measurements ${CERES_LIBRARIES} yaml-cpp)

add_executable(align_spline_to_global_frame src/align_spline_to_global_frame.cpp)
target_link_libraries(align_spline_to_global_frame ${CERES_LIBRARIES} yaml-cpp)

add_executable(optimize_continuous_time src/optimize_continuous_time.cpp)
target_link_libraries(optimize_continuous_time ${CERES_LIBRARIES} yaml-cpp)

add_executable(optimize_vi_continuous_time src/optimize_vi_continuous_time.cpp)
target_link_libraries(optimize_vi_continuous_time ${CERES_LIBRARIES} yaml-cpp)

add_executable(optimize_gi_continuous_time src/optimize_gi_continuous_time.cpp)
target_link_libraries(optimize_gi_continuous_time ${CERES_LIBRARIES} yaml-cpp)

add_executable(optimize_gv_continuous_time src/optimize_gv_continuous_time.cpp)
target_link_libraries(optimize_gv_continuous_time ${CERES_LIBRARIES} yaml-cpp)

add_executable(optimize_discrete_time src/optimize_discrete_time.cpp)
target_link_libraries(optimize_discrete_time ${CERES_LIBRARIES} yaml-cpp)

add_executable(optimize_vi_discrete_time src/optimize_vi_discrete_time.cpp)
target_link_libraries(optimize_vi_discrete_time ${CERES_LIBRARIES} yaml-cpp)

add_executable(optimize_gi_discrete_time src/optimize_gi_discrete_time.cpp)
target_link_libraries(optimize_gi_discrete_time ${CERES_LIBRARIES} yaml-cpp)

add_executable(optimize_gv_discrete_time src/optimize_gv_discrete_time.cpp)
target_link_libraries(optimize_gv_discrete_time ${CERES_LIBRARIES} yaml-cpp)