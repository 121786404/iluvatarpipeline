# 设置CUDA编译器和架构
set(CMAKE_CUDA_ABI_COMPILED TRUE)
set(CMAKE_CUDA_COMPILER_ID "Clang")
set(CMAKE_CUDA_COMPILER "clang++" CACHE STRING "CUDA compiler.")
set(CMAKE_CUDA_HOST_COMPILER "clang++" CACHE STRING "CUDA Host compiler.")
if (NOT CMAKE_CUDA_ARCHITECTURES)
    set(CUDA_ARCH "ivcore11" CACHE STRING "CUDA architecture.")
    set(CMAKE_CUDA_ARCHITECTURES "ivcore11")
endif()
enable_language(CUDA)
# 生成一个静态库或共享库
add_library(ExtraProcess STATIC
    preprocess.cu
    preprocess.h
    IluvatarPreprocess.cpp
)

target_link_libraries(ExtraProcess 
    Common
    ${CUDA_LIBRARIES}
    ${CUDART_LIBRARIES})
