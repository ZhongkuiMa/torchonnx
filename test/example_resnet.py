from torchonnx import TorchONNX

if __name__ == "__main__":
    # The following
    file_path = "../nets/TinyImageNet_resnet_medium_v22_simplified.onnx"
    converter = TorchONNX(verbose=True)
    converter.convert(file_path)
