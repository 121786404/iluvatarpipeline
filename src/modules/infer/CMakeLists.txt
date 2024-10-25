# 生成一个静态库或共享库
find_library(IXRT_LIBRARIES ixrt PATHS ${CUDA_LIB_DIR})
find_library(IXRT_PLUGIN_LIBRARIES  ixrt_plugin PATHS ${CUDA_LIB_DIR})
find_library(OSS_PLUGIN_LIBRARIES  cudnn PATHS ${CUDA_LIB_DIR})

add_library(Infer STATIC
    IluvatarIxRT.cpp
)

target_link_libraries(Infer 
    dl
    ${CUDA_LIBRARIES}
    ${CUDART_LIBRARIES}
    ${IXRT_LIBRARIES}
    ${IXRT_PLUGIN_LIBRARIES}
    ${OSS_PLUGIN_LIBRARIES})