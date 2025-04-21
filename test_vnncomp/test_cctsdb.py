from torchonnx import TorchONNX

if __name__ == "__main__":
    file_path = (
        "../../vnncomp2024_benchmarks/benchmarks/cctsdb_yolo_2023/onnx/"
        "patch-1_v22_simplified.onnx"
    )
    converter = TorchONNX(verbose=True)
    converter.convert(file_path)
