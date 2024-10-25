import onnx
import numpy as np
import argparse

# Argument parser
parser = argparse.ArgumentParser(description="Modify ONNX Reshape Node.")
parser.add_argument('--ori_onnx_path', type=str, required=True, help="Original ONNX file path")
parser.add_argument('--save_onnx_path', type=str, required=True, help="Save path for modified ONNX file")
args = parser.parse_args()

# Load the ONNX model
model = onnx.load(args.ori_onnx_path)
graph = model.graph

# Find the Reshape node and modify its shape
for node in graph.node:
    if node.op_type == "Reshape":
        reshape_input_const_edge = node.input[1]

# Locate the corresponding initializer and modify the shape
for key_val in graph.initializer:
    key = key_val.name
    if key == reshape_input_const_edge:
        shape = np.frombuffer(key_val.raw_data, np.int64)
        tmp = shape.copy()
        tmp[-1] = 1280
        key_val.raw_data = tmp.tobytes()

# Save the modified ONNX model
onnx.save(model, args.save_onnx_path)
