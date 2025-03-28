from torchonnx import TorchONNX

if __name__ == "__main__":
    file_path = "../nets/TinyImageNet_resnet_medium.onnx"
    converter = TorchONNX(verbose=True)
    converter.convert(file_path)
