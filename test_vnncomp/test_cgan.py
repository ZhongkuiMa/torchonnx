from torchonnx import TorchONNX

if __name__ == "__main__":
    file_path = (
        "../../vnncomp2024_benchmarks/benchmarks/cgan_2023/onnx"
        "/cGAN_imgSz32_nCh_3_small_transformer_v22_simplified.onnx"
    )
    converter = TorchONNX(verbose=True)
    converter.convert(file_path)
