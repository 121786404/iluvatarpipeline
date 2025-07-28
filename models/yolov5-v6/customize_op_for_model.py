# !/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from tensorrt.deploy.api import GraphTransform, create_source, create_target

class YoloV5Transform:
    def __init__(self, graph):
        self.t = GraphTransform(graph)
        self.graph = graph

    def AddYoloDecoderOp(self, inputs: list, outputs: list, **attributes):
        print(self.graph.output_names)

        self.t.make_operator(
            "YoloV5Decoder", inputs=inputs, outputs=outputs, **attributes
        )

        return self.graph

    def AddBoxConcatOp(self, inputs: list, outputs, **attributes):
        self.t.make_operator(
            "BoxConcat", inputs=inputs, outputs=outputs, **attributes
        )
        return self.graph

    def AddConcatOp(self, inputs: list, outputs, **attributes):
        self.t.make_operator(
            "Concat", inputs=inputs, outputs=outputs, **attributes
        )
        return self.graph

    def AddGPUNms(self, inputs: list, outputs, **attributes):
        self.t.make_operator(
            "DetectionNMS_IxRT", inputs=inputs, outputs=outputs, **attributes
        )

        for dest, src in zip(outputs, inputs):
            self.t.replace_output(dest, src)
        return self.graph

def customize_ops(graph, fusion_names, include_nms=True):
    t = YoloV5Transform(graph)
    graph = t.AddYoloDecoderOp(
        inputs=[fusion_names[0]],
        outputs=["decoder_32"],
        anchor=[116, 90, 156, 198, 373, 326],
        num_class=80,
        stride=32,
        faster_impl = 1,
    )
    graph = t.AddYoloDecoderOp(
        inputs=[fusion_names[1]],
        outputs=["decoder_16"],
        anchor=[30, 61, 62, 45, 59, 119],
        num_class=80,
        stride=16,
        faster_impl = 1,
    )
    graph = t.AddYoloDecoderOp(
        inputs=[fusion_names[2]],
        outputs=["decoder_8"],
        anchor=[10, 13, 16, 30, 33, 23],
        num_class=80,
        stride=8,
        faster_impl = 1,
    )

    if include_nms:
        graph = t.AddConcatOp(
            inputs=["decoder_32", "decoder_16", "decoder_8"],
            outputs=["all_box_tensor"],
            axis=1
        )

        graph = t.AddGPUNms(
            inputs=["all_box_tensor"],
            outputs=["nms_output0", "int_output1"],
            fIoUThresh=0.65,
            fScoreThresh=0.4,
            nMaxKeep = 1000
        )
        graph.outputs.clear()
        graph.add_output("nms_output0")
        graph.outputs["nms_output0"].dtype = "FLOAT"
        graph.add_output("int_output1")
        graph.outputs["int_output1"].dtype = "INT32"
    else:
        graph = t.AddConcatOp(
            inputs=["decoder_32", "decoder_16", "decoder_8"],
            outputs=["all_box_tensor"],
            axis=1
        )
        graph.outputs.clear()
        graph.add_output("all_box_tensor")
        graph.outputs["all_box_tensor"].dtype = "INT32"
        # graph.outputs.clear()
        # graph.add_output("decoder_8")
        # graph.outputs["decoder_8"].dtype = "FLOAT"
        # graph.add_output("decoder_16")
        # graph.outputs["decoder_16"].dtype = "FLOAT"
        # graph.add_output("decoder_32")
        # graph.outputs["decoder_32"].dtype = "FLOAT"
    return graph

def customize_for_yolov5m(src, dst, fusion_names, include_nms=True):
    graph = create_source(src)()
    graph = customize_ops(graph, fusion_names, include_nms)
    create_target(saved_path=dst).export(graph)
    print("Surged onnx lies on", dst)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", type=str, default="yolov5s_without_decoder.onnx")
    parser.add_argument("--output_model", type=str, default="yolov5s_with_decoder_nms.onnx")
    parser.add_argument("--fusion_names", nargs='+', type=str)
    parser.add_argument("--include_nms", action='store_true')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    customize_for_yolov5m(args.input_model, args.output_model, args.fusion_names, include_nms=args.include_nms)
