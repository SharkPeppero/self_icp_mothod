cmake_minimum_required(VERSION 3.8)
project(eigen_icp)

########################################## 编译器设置 ##########################################
# 检查当前C++编译器是否为 GNU Compiler Collection 或者 检查 CMAKE_CXX_COMPILER_ID 是否匹配字符串 "Clang"
# 如果条件为真（即使用的是 GCC 或 Clang 编译器）
# -Wall 启用了大多数警告消息，-Wextra 启用了额外的警告消息，-Wpedantic 启用了更严格的警告
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Default to C99
if(NOT CMAKE_C_STANDARD)
    set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif()

set(CMAKE_BUILD_TYPE "Debug")

include_directories(
        "${PROJECT_SOURCE_DIR}/include"
)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})
if(${Eigen3_FOUND})
    message(VERBOSE "Eigen3's Version is:${Eigen3_VERSION}")
endif()

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
if(${PCL_FOUND})
    message(VERBOSE "PCL's Version is:${PCL_VERSION}")
endif()

find_package(TBB REQUIRED)

add_executable(eigen_icp_node
    main.cpp
)
target_link_libraries(eigen_icp_node
        ${PCL_LIBRARIES}
        pthread
        TBB::tbb)



