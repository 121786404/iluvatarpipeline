# ONNX2IXRT
0. 安装ixrt python 和 cuda_python
```bash
pip install ../../3rdparty/ixrt/ixrt-0.9.1+corex.4.1.2.20240906.161-cp38-cp38-linux_aarch64.whl -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/ 
pip install cuda_python-11.8.0+corex.4.1.2.20240906.161-cp38-cp38-linux_aarch64.whl -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/ 
```

1. download models 

```bash
wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.onnx
```

2. 简化模型和修改输入大小

需要安装onnxsim
```bash
pip install onnxsim -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/ 
onnxsim ./yolov5s.onnx ./yolov5s_sim.onnx --overwrite-input-shape "images:-1,3,640,640"
```

3. 裁剪和填加ixrt-nms算子

3-1. 裁剪模型, 4.3.0之后版本主要修改:cut位置前提至最后一个Conv结束，-1x512x20x20,-1x512x40x40,-1x512x80x80,
```bash
python3 cut_model.py \
        --input_model ./yolov5s_sim.onnx \
        --output_model ./yolov5s_sim_cut.onnx  \
        --input_names "images" \
        --output_names "326" "474" "622"
```

3-2. 填加ixrt-nms算子, 4.3.0之后版本主要修改:nms plugin名字修改，由原来的`NMS`替换为`DetectionNMS_IxRT`

请注意将 AddYoloDecoderOp时需要确认anchor与相应的大小的层对应
例如cut后的三个结果中，大小为 ?x255x80x80 -> 对应的anchor参数如下
``` bash
    graph = t.AddYoloDecoderOp(
        inputs=[fusion_names[2]],
        outputs=["decoder_8"],
        anchor=[10, 13, 16, 30, 33, 23],
        num_class=80,
        stride=8,
        faster_impl = 1,
    )
```

```bash
python3 ./customize_op_for_model.py \
        --input_model ./yolov5s_sim_cut.onnx \
        --output_model ./yolov5s_sim_cut_withnms.onnx  \
        --fusion_names "622" "474" "326" \
        --include_nms
```

4. 转换ixrt engine
```bash
ixrtexec --onnx ./yolov5s_sim_cut_withnms.onnx \
        --precision fp16 \
        --min_shape images:1x3x640x640 \
        --opt_shape images:16x3x640x640 \
        --max_shape images:32x3x640x640 \
        --plugins /usr/local/corex/lib/libixrt_plugin.so \
        --save_engine ./yolov5s_sim_cut_withnms.engine 
```