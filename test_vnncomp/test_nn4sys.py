from torchonnx import TorchONNX

if __name__ == "__main__":
    file_path = (
        "../../vnncomp2024_benchmarks/benchmarks/nn4sys_2023/onnx/"
        "mscn_2048d_dual_v22_simplified.onnx"
    )
    converter = TorchONNX(verbose=True)
    converter.convert(file_path)
