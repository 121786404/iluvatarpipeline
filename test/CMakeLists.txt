# 添加测试案例

add_executable(pipeline1 Pipeline_OneStage.cpp)
target_link_libraries(pipeline1 PipelineLib)

add_executable(pipeline2 Pipeline_TwoStage.cpp)
target_link_libraries(pipeline2 PipelineLib)
