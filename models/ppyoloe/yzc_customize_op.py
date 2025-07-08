from onnx import helper, numpy_helper, TensorProto
import argparse
import sys
import onnx


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--src", help="The exported to ONNX YOLOX-m", type=str,
    )
    parser.add_argument(
        "-o",
        "--dest",
        help="The output ONNX model file to write",
        type=str,
    )
    parser.add_argument("--fusion_names", nargs='+', type=str)
    parser.add_argument(
        "-m",
        "--max_output_boxes",
        help="Max output boxes for the NMS operation",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "-st",
        "--score_threshold",
        help="The scalar threshold for score",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "-it",
        "--iou_threshold",
        help="The scalar threshold for IOU",
        type=float,
        default=0.45,
    )

    args = parser.parse_args()
    if not all(
        [
            args.src,
            args.dest,
            args.max_output_boxes,
            args.score_threshold,
            args.iou_threshold,
        ]
    ):
        parser.print_help()
        print(
            "\nThese arguments are required: --src --dest --max_output_boxes --score_threshold --iou_threshold"
        )
        sys.exit(1)
    print(args)
    return args

if __name__ == "__main__":
    config = parse_args()
    ppyoloe = onnx.load(config.src)
    graph = ppyoloe.graph
    nodes = graph.node

    new_shape = [-1, 1, 10710, 4]
    reshape_tensor = helper.make_tensor(
        name = "reshape_shape",
        data_type=TensorProto.INT64,
        dims = [len(new_shape)],
        vals = new_shape
    )

    # 创建一个常量节点，包含shape信息
    shape_node = helper.make_node(
        "Constant",  # 节点类型
        inputs=[],   # Constant节点没有输入
        outputs=["reshape_shape"],  # 输出tensor的名字
        value=reshape_tensor  # 常量值
    )

    # 2. 找到Mul.157的输出名
    mul_output_name = config.fusion_names[0]

    for node in nodes:
        if node.name == mul_output_name:
            out_name = node.output[0]
            break
    else:
        print(f"未找到节点: {mul_output_name}")
        sys.exit(1)


    # 3. 创建Reshape节点
    reshape_output_name = "output_reshape"
    reshape_node = helper.make_node(
        "Reshape",
        inputs=[out_name, "reshape_shape"],
        outputs=[reshape_output_name],
        name="reshape_op"
    )

    # 新增：创建Transpose节点
    transpose_output_name = "output_transpose"
    transpose_node = helper.make_node(
        "Transpose",
        inputs=[reshape_output_name],
        outputs=[transpose_output_name],
        name="transpose_op",
        perm=[0, 3, 1, 2]
    )

    # 4. 替换所有以mul_output_name为输入的节点为reshape_output_name
    for node in graph.node:
        for i, inp in enumerate(node.input):
            if inp == mul_output_name:
                node.input[i] = reshape_output_name

    # 新增：替换所有以reshape_output_name为输入的节点为transpose_output_name
    for node in graph.node:
        for i, inp in enumerate(node.input):
            if inp == reshape_output_name:
                node.input[i] = transpose_output_name

    # 5. 如果graph.output用到了mul_output_name，也要替换
    for output in graph.output:
        if output.name == mul_output_name:
            output.name = transpose_output_name
            # 清空并重设shape
            while len(output.type.tensor_type.shape.dim) > 0:
                del output.type.tensor_type.shape.dim[0]
            # 由于transpose后的shape未知，这里不设置具体shape

    # 6. 插入新节点
    insert_idx = None
    for idx, node in enumerate(graph.node):
        if node.name == mul_output_name:
            insert_idx = idx + 1
            break
    
    if insert_idx is not None:
        graph.node.insert(insert_idx, shape_node)
        graph.node.insert(insert_idx + 1, reshape_node)
        graph.node.insert(insert_idx + 2, transpose_node)

        # 新增：插入NMS_IXRT节点
        nms_output_names = [
            "num_detections",
            "detection_boxes",
            "detection_scores",
            "detection_classes"
        ]
        nms_node = helper.make_node(
            "NMS_IXRT",
            inputs=[transpose_output_name, config.fusion_names[1]],
            outputs=nms_output_names,
            name="nms",
            background_class=-1,
            iou_threshold=config.iou_threshold,
            max_output_boxes=1000,
            score_threshold=config.score_threshold,
            share_location=1
        )
        graph.node.insert(insert_idx + 3, nms_node)

        # 新增：设置graph.output为NMS输出，类型为float32
        while len(graph.output) > 0:
            del graph.output[0]
        for out_name in nms_output_names:
            output_tensor = helper.make_tensor_value_info(
                out_name, TensorProto.FLOAT, None
            )
            graph.output.append(output_tensor)

        # 新增：为Reshape和Transpose输出添加value_info（定义为tensor）
        reshape_value_info = helper.make_tensor_value_info(
            reshape_output_name, TensorProto.FLOAT, [-1, 1, 10710, 4]
        )
        transpose_value_info = helper.make_tensor_value_info(
            transpose_output_name, TensorProto.FLOAT, [-1, 4, 1, 10710]
        )
        graph.value_info.append(reshape_value_info)
        graph.value_info.append(transpose_value_info)
    else:
        # 如果没找到Mul.156，直接加到最后
        print("insert op fail")

    # 7. 保存模型
    onnx.save(ppyoloe, config.dest)
    print(f"模型已保存到: {config.dest}")







