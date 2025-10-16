# yolov8m模型
## 下载模型

yolov8官方源码路径：`https://github.com/Pertical/YOLOv8/tree/main`
```bash
git clone https://github.com/Pertical/YOLOv8.git
```

# Onnx2Ixrt

## 简化模型和修改输入大小

需要安装onnxsim
```bash
pip install onnxsim
onnxsim ./yolov8m.onnx ./yolov8m_sim.onnx --overwrite-input-shape "image:-1,3,640,640"
```

## 添加xywh2xyxy和填加ixrt-nms算子

1. 模型添加xywh2xyxy
```bash
python3 ./add_xywh2xyxy_onnx.py

```

2. 填加ixrt-nms算子

```bash
python3 ./customize_op_for_model.py \
        --src ./yolov8m_sim_xyxy.onnx \
        --dest ./yolov8m_sim_xyxy_withnms.onnx \
        --fusion_names "output_boxes" "output_scores" \
        --max_output_boxes 1000 \
        --score_threshold 0.5 \
        --iou_threshold 0.5

```

## 转换ixrt engine
```bash
ixrtexec --onnx ./yolov8m_sim_xyxy_withnms.onnx \
        --precision fp16 \
        --min_shape image:1x3x640x640 \
        --opt_shape image:16x3x640x640 \
        --max_shape image:32x3x640x640 \
        --plugins /usr/local/corex/lib/liboss_ixrt_plugin.so \
        --save_engine ./yolov8m_sim_xyxy_withnms.engine 
```

## python测试脚本
```bash
python3 ./inference_ixrt_dyn.py \
        --model_engine ./yolov8m_sim_xyxy_withnms.engine \
        --input_image ./dog_640.jpg
```