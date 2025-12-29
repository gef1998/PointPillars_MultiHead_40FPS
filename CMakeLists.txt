cmake_minimum_required(VERSION 3.17)
set(CATKIN_ENABLE_TESTING OFF)
set(GTEST_FOUND OFF CACHE BOOL "" FORCE)


set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0042 NEW)
project(pointpillars LANGUAGES CXX CUDA)


option(BUILD_SHARED_LIBS "Build as a shared library"  ON)
option(USE_DOUBLE        "Use double as kernel_type" OFF)
option(TEST              "BUILD TEST"                 ON)
option(BENCH             "BUILD BENCH"               OFF)

if(USE_DOUBLE)
    message("Use double as kernel_type")
endif()
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()
############################################################
# add ROS nodes
#############################################################
set(PYTHON_EXECUTABLE /usr/bin/python) 
find_package(catkin REQUIRED COMPONENTS
    roscpp
    sensor_msgs
    autoware_msgs
    geometry_msgs
    tf
)
include_directories(${catkin_INCLUDE_DIRS})


add_executable(pc_infer
    nodes/pc_infer.cc
    nodes/det3d_publisher.cc
)
target_link_libraries(pc_infer
    ${PROJECT_LIB_NAME}
    ${LINK_LIBRARY}
    ${catkin_LIBRARIES}
    pointpillars
)
target_include_directories(pc_infer PRIVATE
    ${catkin_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/pointpillars/include
)


############################################################
# add 3th_part library
############################################################

find_package(yaml-cpp REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIRS})

find_package(Threads)
find_package(OpenMP REQUIRED)
add_definitions("-DELPP_NO_LOG_TO_FILE")

find_package(CUDA REQUIRED QUIET)
include_directories(${CUDA_INCLUDE_DIRS})
if (CUDA_VERSION VERSION_LESS "11.0")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11 -Wno-deprecated-gpu-targets)
else ()
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++14 -Wno-deprecated-gpu-targets)
endif ()

find_library(NVINFER NAMES nvinfer)
find_library(NVPARSERS NAMES nvparsers)
find_library(NVONNXPARSERS NAMES nvonnxparser)

list(APPEND LINK_LIBRARY
            ${CUDA_cusparse_LIBRARY}
            ${NVINFER}
            ${NVONNXPARSERS}
            ${CUDA_LIBRARIES}
            ${CUDA_CUBLAS_LIBRARIES}
            ${CUDA_curand_LIBRARY}            
            ${CUDA_cusparse_LIBRARY}

            yaml-cpp
            )

# Correct error for GCC 5 and CUDA 7.5
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DYAML_BUILD_SHARED_LIBS=ON -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES ")

if (CMAKE_VERSION VERSION_LESS "3.1")
    add_compile_options("-std=c++11 -fext-numeric-literals")
else ()
    set(CMAKE_CXX_STANDARD 11)
endif ()

set(COMMON_INCLUDES ${CMAKE_CURRENT_BINARY_DIR})

############################################################
# Create a library in subdirectory
############################################################
set(PROJECT_LIB_NAME ${PROJECT_NAME})
include_directories(${COMMON_INCLUDES})
add_subdirectory(${PROJECT_SOURCE_DIR}/pointpillars)


############################################################
# Create an executable
############################################################
if (TEST)
    message("Building tests")
    enable_testing()
    add_subdirectory(test)
endif ()

message(STATUS "")
message(STATUS "BUILD SUMMARY")
message(STATUS "  CMAKE_GENERATOR      : ${CMAKE_GENERATOR}")
message(STATUS "  Compiler ID          : ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  Build type           : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Build shared libs    : ${BUILD_SHARED_LIBS}")
message(STATUS "  Use double for kernel: ${USE_DOUBLE}")
message(STATUS "  Build tests          : ${TEST}")
message(STATUS "")
