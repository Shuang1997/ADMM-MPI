cmake_minimum_required(VERSION 2.8)

project(ADMM_MPI)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-std=c++11 -O0")

set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpicxx)

find_package(MPI REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
    "include"
    ${MPI_INCLUDE_PATH}
    ${Eigen3_INCLUDE_DIR})

add_executable(parallel_admm src/main.cpp)
target_link_libraries(parallel_admm ${MPI_LIBRARIES})

#if(MPI_COMPILE_FLAGS)
#  set_target_properties(hello PROPERTIES
#    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
#endif()

#if(MPI_LINK_FLAGS)
#  set_target_properties(hello PROPERTIES
#    LINK_FLAGS "${MPI_LINK_FLAGS}")
#endif()
