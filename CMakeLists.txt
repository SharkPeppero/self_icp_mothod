cmake_minimum_required(VERSION 3.8)
project(eigen_icp)

########################################## 编译器设置 ##########################################
# 检查当前C++编译器是否为 GNU Compiler Collection 或者 检查 CMAKE_CXX_COMPILER_ID 是否匹配字符串 "Clang"
# 如果条件为真（即使用的是 GCC 或 Clang 编译器）
# -Wall 启用了大多数警告消息，-Wextra 启用了额外的警告消息，-Wpedantic 启用了更严格的警告
if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif ()

# Default to C99
if (NOT CMAKE_C_STANDARD)
    set(CMAKE_C_STANDARD 99)
endif ()

# Default to C++14
if (NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 14)
endif ()

# set save reference path
add_definitions(-DDEBUG_PATH="${CMAKE_CURRENT_SOURCE_DIR}/debug_path/")
message(STATUS "DEBUG_PATH: ${CMAKE_CURRENT_SOURCE_DIR}/debug_path/")

set(CMAKE_BUILD_TYPE "Debug")

# set cuda options
set(USE_CUDA OFF)
if (USE_CUDA)
    MESSAGE(STATUS "USE CUDA to do registration")
    add_definitions(-DUSE_CUDA)

    set(CMAKE_NVCC_FLAGS "CMAKE_NVCC_FLAGS -std=c++14") # 设置NVCC编译器的编译版本

    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")

    if ("${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
        set(CUDA_LINK_DIRECTORIES_PATH "/home/xu/3rdparty/cuda_11.8/targets/x86_64-linux/lib")
        message(STATUS "CUDA's link direction is: ${CUDA_LINK_DIRECTORIES_PATH}")
        link_directories(usr/local/cuda-10.2/targets/aarch64-linux/lib)
        add_definitions(-D BUILD_ON_XAVIER)
    elseif ("${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
#        set(CUDA_LINK_DIRECTORIES_PATH "/home/xu/3rdparty/cuda_11.8/targets/x86_64-linux/lib")
        set(CUDA_LINK_DIRECTORIES_PATH "usr/local/cuda-10.2/targets/aarch64-linux/lib")
        message(STATUS "CUDA link direction is: ${CUDA_LINK_DIRECTORIES_PATH}")
        link_directories(${CUDA_LINK_DIRECTORIES_PATH})
        add_definitions(-D BUILD_ON_X86)
    endif ()

    #   NVIDIA Performance Primitives library (NPP)是一系列为加速进程的库函数，NPP的原始目标是针对图像和视频进程，而现在以被开发工程师应用于其它很多领域，包括信号处理。
    #   与同级别的纯CPU 函数相比，这些函数最高可实现5 - 10倍性能提升。利用NPP，
    #   开发者能够利用 (CUDA4.1中)2000多个图像处理与信号处理基元，在数小时之内即可实现应用程序的大幅性能提升。
    #   无论用GPU加速的版本代替CPU基元还是将NPP基元与现有的GPU加速流水线相结合，NPP都能够实现极高的性能，同时可缩短开发时间。
    # 详细解释的链接: https://blog.csdn.net/weixi234/article/details/137580705?ops_request_misc=&request_id=&biz_id=102&utm_term=npp&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-4-137580705.142^v101^pc_search_result_base4&spm=1018.2226.3001.4187
    find_cuda_helper_libs(nppc)
    message(STATUS "nppc: ${CUDA_nppc_LIBRARY}")
    find_cuda_helper_libs(nppig)
    message(STATUS "nppig: ${CUDA_nppig_LIBRARY}")
    find_cuda_helper_libs(nppidei)
    message(STATUS "nppidei: ${CUDA_nppidei_LIBRARY}")
    set(CUDA_npp_LIBRARY "${CUDA_nppc_LIBRARY};${CUDA_nppig_LIBRARY};${CUDA_nppidei_LIBRARY}")
endif (USE_CUDA)

include_directories(
        "${PROJECT_SOURCE_DIR}/include"
)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})
if (${Eigen3_FOUND})
    message(VERBOSE "Eigen3's Version is:${Eigen3_VERSION}")
endif ()

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
if (${PCL_FOUND})
    message(VERBOSE "PCL's Version is:${PCL_VERSION}")
endif ()

option(USE_TBB "USE TBB Or Not" Off)
if (USE_TBB)
    find_package(TBB REQUIRED)
    #    TBB::tbb
endif (USE_TBB)

add_executable(self_icp_node
        main.cpp
)
target_link_libraries(self_icp_node
        ${PCL_LIBRARIES}
        pthread)



