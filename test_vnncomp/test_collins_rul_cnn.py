from torchonnx import TorchONNX

if __name__ == "__main__":
    file_path = (
        "../../vnncomp2024_benchmarks/benchmarks/collins_rul_cnn_2023/onnx"
        "/NN_rul_full_window_40_v22_simplified.onnx"
    )
    converter = TorchONNX(verbose=True)
    converter.convert(file_path)
