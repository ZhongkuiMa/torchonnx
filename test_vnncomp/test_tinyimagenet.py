from torchonnx import TorchONNX

if __name__ == "__main__":
    file_path = (
        "../../vnncomp2024_benchmarks/benchmarks/tinyimagenet/onnx/"
        "TinyImageNet_resnet_medium_v22_simplified.onnx"
    )
    converter = TorchONNX(verbose=True)
    converter.convert(file_path)
