# 查找第三方库
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/ffmpeg/include)
#set(FFMPEG_LIB_PATH ${CMAKE_SOURCE_DIR}/3rdparty/ffmpeg/lib)
set(FFMPEG_LIB_PATH ${CMAKE_SOURCE_DIR}/3rdparty/ffmpeg/lib/${CMAKE_SYSTEM_PROCESSOR})


find_library(FFMPEG_AVCODEC avcodec PATHS ${FFMPEG_LIB_PATH} NO_DEFAULT_PATH)
find_library(FFMPEG_AVFORMAT avformat PATHS ${FFMPEG_LIB_PATH} NO_DEFAULT_PATH)
find_library(FFMPEG_AVUTIL avutil PATHS ${FFMPEG_LIB_PATH} NO_DEFAULT_PATH)
find_library(FFMPEG_SWSCALE swscale PATHS ${FFMPEG_LIB_PATH} NO_DEFAULT_PATH)
find_library(FFMPEG_SWRESAMPLE swresample PATHS ${FFMPEG_LIB_PATH} NO_DEFAULT_PATH)

find_library(NVCUVID_LIBRARIES nvcuvid PATHS ${CUDA_LIB_DIR} NO_DEFAULT_PATH)
find_library(NVJPEG_LIBRARIES  nvjpeg PATHS ${CUDA_LIB_DIR} NO_DEFAULT_PATH)

# 生成一个静态库或共享库
add_library(Codec STATIC
    FFmpegDemuxer.cpp
    IluvatarJpegCodec.cpp
    IluvatarVideoDecoder.cpp
)

# 链接第三方库和Common库
target_link_libraries(Codec 
    Common 
    ${FFMPEG_AVCODEC}
    ${FFMPEG_AVFORMAT}
    ${FFMPEG_AVUTIL}
    ${FFMPEG_SWSCALE}
    ${FFMPEG_SWRESAMPLE}
    ${CUDA_LIBRARIES}
    ${CUDART_LIBRARIES}
    ${NVJPEG_LIBRARIES}
    ${NVCUVID_LIBRARIES}
    z
    lzma
    bz2
)
