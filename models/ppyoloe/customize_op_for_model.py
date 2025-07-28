import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference
import numpy as np
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--src", help="The exported to ONNX YOLOX-m", type=str,
    )
    parser.add_argument(
        "-o", "--dest", help="The output ONNX model file to write", type=str,
    )
    parser.add_argument(
        "--fusion_names", nargs='+', type=str, help="Two output node names: [Mul.90, Concat.29]"
    )
    parser.add_argument(
        "-m", "--max_output_boxes", help="Max output boxes for the NMS operation", type=int, default=1000,
    )
    parser.add_argument(
        "-st", "--score_threshold", help="The scalar threshold for score", type=float, default=0.7,
    )
    parser.add_argument(
        "-it", "--iou_threshold", help="The scalar threshold for IOU", type=float, default=0.45,
    )

    args = parser.parse_args()
    if not all([args.src, args.dest, args.fusion_names]):
        parser.print_help()
        print(
            "\nThese arguments are required: --src --dest --fusion_names"
        )
        sys.exit(1)
    print(args)
    return args

def modify_model(args):
    model = onnx.load(args.src)
    graph = model.graph

    mul_output_name = args.fusion_names[0]
    concat_output_name = args.fusion_names[1]

    # 1. Reshape [-1, 1, 8400, 4]
    reshape_shape_name = "reshape_shape_const"
    reshape_output_name = "reshape_output"
    reshape_shape = np.array([-1, 1, 8400, 4], dtype=np.int64)
    reshape_shape_tensor = numpy_helper.from_array(reshape_shape, name=reshape_shape_name)
    graph.initializer.append(reshape_shape_tensor)

    reshape_node = helper.make_node(
        "Reshape",
        inputs=[mul_output_name, reshape_shape_name],
        outputs=[reshape_output_name],
        name="reshape_for_nms"
    )

    # 2. Transpose perm=[0, 3, 1, 2]
    transpose_output_name = "transpose_output"
    transpose_node = helper.make_node(
        "Transpose",
        inputs=[reshape_output_name],
        outputs=[transpose_output_name],
        perm=[0, 3, 1, 2],
        name="transpose_for_nms"
    )

    # 3. NMS_IXRT 节点
    nms_output_names = [
            "num_detections",
            "detection_boxes",
            "detection_scores",
            "detection_classes"
        ]
    nms_node = helper.make_node(
        "NMS_IXRT",
        inputs=[transpose_output_name, concat_output_name],
        outputs=nms_output_names,
        name="nms",
        background_class=-1,
        iou_threshold=args.iou_threshold,
        max_output_boxes=args.max_output_boxes,
        score_threshold=args.score_threshold,
        share_location=1
    )

    # 添加节点
    graph.node.extend([reshape_node, transpose_node, nms_node])

    # 设置模型输出（全部4个 NMS 输出）
    graph.output.clear()
    graph.output.extend([
        helper.make_tensor_value_info(nms_output_names[0], TensorProto.FLOAT, None),   # num_detections
        helper.make_tensor_value_info(nms_output_names[1], TensorProto.FLOAT, None),   # detection_boxes
        helper.make_tensor_value_info(nms_output_names[2], TensorProto.FLOAT, None),   # detection_scores
        helper.make_tensor_value_info(nms_output_names[3], TensorProto.FLOAT, None),   # detection_classes
    ])

    # 可选：shape 推理
    model = shape_inference.infer_shapes(model)

    # 保存模型
    onnx.save(model, args.dest)
    print(f"Modified model saved to: {args.dest}")

if __name__ == "__main__":
    args = parse_args()
    modify_model(args)
