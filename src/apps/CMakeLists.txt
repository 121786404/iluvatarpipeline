# 添加 pipeline 静态库
add_library(PipelineLib STATIC 
    base_processor.cpp
    pipeline_one_stage.cpp
    pipeline_two_stage.cpp
    )
target_link_libraries(PipelineLib
    Common
    Codec
    ExtraProcess
    Infer
    ${CUDA_LIBRARIES}
    ${CUDART_LIBRARIES}
)