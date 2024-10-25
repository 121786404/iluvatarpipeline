# 查找第三方库
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/json/include)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/spdlog/include)

# 生成一个静态库或共享库
add_library(Common STATIC
    surface/memoryInterface.cpp
    util/Logger.cpp
)

# 链接第三方库
target_link_libraries(Common 
    ${CMAKE_SOURCE_DIR}/3rdparty/spdlog/lib/${CMAKE_SYSTEM_PROCESSOR}/libspdlog.a
    pthread
    dl
    ${CUDA_LIBRARIES}
    ${CUDART_LIBRARIES}
)
