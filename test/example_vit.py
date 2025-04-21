from torchonnx import TorchONNX

if __name__ == "__main__":
    file_path = "../nets/ibp_3_3_8_v22_simplified.onnx"
    converter = TorchONNX(verbose=True)
    converter.convert(file_path)
