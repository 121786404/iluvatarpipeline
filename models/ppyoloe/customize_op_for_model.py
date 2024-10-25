# !/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import sys

from tensorrt.deploy.api import DataType, GraphTransform, create_source, create_target
import onnx

class PPYoloETransform:
    def __init__(self, graph):
        self.t = GraphTransform(graph)
        self.graph = graph

    def AddVar(self, name: str, value: None, **attributes):
        self.t.make_variable(
            name=name, value=value, **attributes
        )
        return self.graph
    
    def AddReshape(self, name: str, inputs: list, outputs: list, **attributes):
        self.t.make_operator(
            "Reshape", name=name, inputs=inputs, outputs=outputs, **attributes
        )
        return self.graph
    
    def AddTranspose(self, name: str, inputs: list, outputs: list, **attributes):
        self.t.make_operator(
            "Transpose", name=name, inputs=inputs, outputs=outputs, **attributes
        )
        return self.graph

    def AddNMSOp(self, name: str, inputs: list, outputs: list, **attributes):
        self.t.make_operator(
            "NMS_IXRT", name=name, inputs=inputs, outputs=outputs, **attributes
        )
        for var_name in outputs:
            self.t.add_output(var_name)
            self.t.get_variable(var_name).dtype = "FLOAT"
        return self.graph

    def DropUselessOutputs(
        self,
        drop_list
    ):
        for var_name in drop_list:
            self.t.delete_output(var_name)
        return self.graph

    def Cleanup(
        self,
    ):
        self.t.cleanup()

        return self.graph


def add_yolox_postprocess_nodes(graph, config):
    t = PPYoloETransform(graph)
    graph = t.AddVar(
        name="reshape_var",
        value=[-1,1,8400,4]
    )
    
    graph = t.AddReshape(
        name="reshape_nms",
        inputs=[config.fusion_names[0],"reshape_var"],
        outputs=["reshape_out"]
    )
    
    graph = t.AddTranspose(
        name="transpose_nms",
        inputs=["reshape_out"],
        outputs=["transpose_out"],
        perm=[0,3,1,2],
    )

    graph = t.AddNMSOp(
        name="nms",
        inputs=["transpose_out", config.fusion_names[1]],
        outputs=[
            "num_detections",
            "detection_boxes",
            "detection_scores",
            "detection_classes",
        ],
        share_location=1,
        iou_threshold=config.iou_threshold,
        score_threshold=config.score_threshold,
        max_output_boxes=config.max_output_boxes,
        background_class=-1,
    )

    graph = t.DropUselessOutputs(config.fusion_names)
    graph = t.Cleanup()

    return graph

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
    graph = create_source(config.src)()
    graph = add_yolox_postprocess_nodes(graph, config)
    create_target(saved_path=config.dest).export(graph)
    
    model = onnx.load(config.dest)
    print("Added post-processing layers onnx lies on", config.dest)
