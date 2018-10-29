cmake_minimum_required(VERSION 2.8)
project(mobileNet)

#set(inference_VERSION_MAJOR 2)
#set(inference_VERSION_MINOR 1)

#set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(BUILD_DEPS "YES" CACHE BOOL "If YES, will install dependencies into sandbox.  Automatically reset to NO after dependencies are installed.")

set(PROJECT_OUTPUT_DIR  ${PROJECT_BINARY_DIR}/build)
set(PROJECT_INCLUDE_DIR ${PROJECT_OUTPUT_DIR}/include)

file(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR})
file(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)

message("The runtime libraries are included in ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message("The library files are included in ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")

message("-- system arch:  ${CMAKE_SYSTEM_PROCESSOR}")
message("-- output path:  ${PROJECT_OUTPUT_DIR}")

find_package(CUDA)
find_package(OpenCV REQUIRED)
message(" -- CUDA and Opencv Found ")
message(" -- opencv_version  "${OpenCV_VERSION})


set(CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};--disable-warnings;
        -O3
        -gencode arch=compute_30,code=sm_30
        -gencode arch=compute_35,code=sm_35
        -gencode arch=compute_50,code=sm_50
        -gencode arch=compute_50,code=compute_50
        -gencode arch=compute_52,code=sm_52
        -gencode arch=compute_61,code=sm_61
	-gencode arch=compute_62,code=sm_62
        )

file(GLOB cudaSources util/cuda/*.cu)
file(GLOB cudaIncludes util/cuda/*.h)

file(GLOB sources *.cu *.cpp util/*.cpp util/cuda/*.cu)
file(GLOB includes util/*.h util/cuda/*.h)

include_directories(${PROJECT_INCLUDE_DIR}/util)
include_directories(${PROJECT_BINARY_DIR}/util)
include_directories(${OpenCV_INCLUDE_DIRS})


link_directories(${OpenCV_LIBRARY_DIRS})

cuda_add_library(inferLib SHARED ${sources})
target_link_libraries(inferLib /usr/lib/aarch64-linux-gnu/libnvcaffe_parser.so)
target_link_libraries(inferLib /usr/lib/aarch64-linux-gnu/libnvinfer.so)
target_link_libraries(inferLib /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so)
target_link_libraries(inferLib /usr/lib/aarch64-linux-gnu/libnvparsers.so)

# transfer all headers to the include directory
foreach(include ${includes})
    message("-- Copying ${include}")
    configure_file(${include} ${PROJECT_INCLUDE_DIR} COPYONLY)
endforeach()

## install
foreach(include ${includes})
    install(FILES "${include}" DESTINATION include/inferLib)
endforeach()

add_executable(mobileNet main.cpp )
target_link_libraries(mobileNet inferLib ${OpenCV_LIBS})
