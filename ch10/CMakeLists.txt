cmake_minimum_required(VERSION 2.8)
project(pose_graph)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# sophus
find_package(Sophus REQUIRED)
find_package(fmt REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})


# g2o 
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})
set(G2O_LIBS g2o_core g2o_stuff g2o_types_slam3d)

add_executable(pose_graph_g2o_SE3 pose_graph_g2o_SE3.cpp)
target_link_libraries(pose_graph_g2o_SE3
        ${G2O_LIBS}
        ${CHOLMOD_LIBRARIES}
        )

add_executable(pose_graph_g2o_lie pose_graph_g2o_lie_algebra.cpp)
target_link_libraries(pose_graph_g2o_lie
        ${G2O_LIBS}
        fmt::fmt
        ${CHOLMOD_LIBRARIES}
        ${Sophus_LIBRARIES}
        )

