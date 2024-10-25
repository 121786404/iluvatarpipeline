# Paddle2Onnx
## 下载模型

paddle官方模型路径：`https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/models/ImageNet1k/PP-LCNet.md`
```bash
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPLCNet_x1_0_infer.tar
```

## 转换onnx

参考paddle官方文档paddle2onnx `https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/deployment/image_classification/paddle2onnx.md`

# Onnx2Ixrt

## 简化模型和修改输入大小

需要安装onnxsim
```bash
pip install onnxsim
onnxsim ./PPLCNet_x1_0_infer.onnx ./PPLCNet_x1_0_infer_sim.onnx --overwrite-input-shape "x:-1,3,224,224"
```

由于sim过程中reshape的大小没有指定，会影响推理结果，使用`modify_reshape.py`修改reshape参数
```bash
python3 ./modify_reshape.py \
        --ori_onnx_path ./PPLCNet_x1_0_infer_sim.onnx \
        --save_onnx_path ./PPLCNet_x1_0_infer_sim_reshape.onnx
```

## 转换ixrt engine
```bash
ixrtexec --onnx ./PPLCNet_x1_0_infer_sim_reshape.onnx \
        --precision fp16 \
        --min_shape x:1x3x224x224 \
        --opt_shape x:32x3x224x224 \
        --max_shape x:64x3x224x224 \
        --save_engine ./PPLCNet_x1_0_infer_sim_reshape.engine 
```