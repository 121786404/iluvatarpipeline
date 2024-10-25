cmake_minimum_required(VERSION 3.2)

# 设置C++标准和调试选项
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Release)

# 项目名称和语言
set(PROJ IluvatarPipeLine)
project(${PROJ})

set(CUDA_LIB_DIR /usr/local/corex/lib64)
set(CUDA_INC_DIR /usr/local/corex/include)
include_directories(${CUDA_INC_DIR})
find_library(CUDA_LIBRARIES cuda PATHS ${CUDA_LIB_DIR})
find_library(CUDART_LIBRARIES cudart PATHS ${CUDA_LIB_DIR})

# 包含FFmpeg库
set(FFMPEG_HEADER_DIR ${PROJECT_SOURCE_DIR}/3rdparty/ffmpeg/include)
include_directories(${FFMPEG_HEADER_DIR})

# 包含nlohmann/json
set(JSON_HEADER_DIR ${PROJECT_SOURCE_DIR}/3rdparty/json/include)
include_directories(${JSON_HEADER_DIR})

# 包含spdlog
include_directories(${PROJECT_SOURCE_DIR}/3rdparty/spdlog/include)

# 包含头文件目录
include_directories(inc)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
add_subdirectory(src/common)
add_subdirectory(src/modules/codec)
add_subdirectory(src/modules/extraProcess)
add_subdirectory(src/modules/infer)
add_subdirectory(src/apps)
add_subdirectory(test)