#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

void print_shape(const std::vector<std::int64_t>& shape) {
    std::cout << '[';
    for (std::size_t dimension = 0; dimension < shape.size(); ++dimension) {
        if (dimension > 0) {
            std::cout << ", ";
        }
        std::cout << shape[dimension];
    }
    std::cout << ']';
}

std::size_t element_count(const std::vector<std::int64_t>& shape) {
    return std::accumulate(
        shape.begin(), shape.end(), std::size_t{1},
        [](std::size_t count, std::int64_t dimension) {
            return count * static_cast<std::size_t>(dimension);
        });
}

}  // namespace

int main(int argc, char* argv[]) {
    const std::filesystem::path model_path =
        argc > 1 ? argv[1] : "models/yolo11n/yolo11n.onnx";

    std::cout << "ONNX Runtime version: " << Ort::GetVersionString() << '\n';
    std::cout << "Model: " << model_path.string() << '\n';

    if (!std::filesystem::is_regular_file(model_path)) {
        std::cerr << "Error: model file not found.\n";
        std::cerr << "Run this program from the repository root or pass the "
                     "model path as its first argument.\n";
        return 1;
    }

    try {
        Ort::Env environment{ORT_LOGGING_LEVEL_WARNING, "yolo11n_verifier"};
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);

        Ort::Session session{
            environment, model_path.c_str(), session_options};
        Ort::AllocatorWithDefaultOptions allocator;

        auto input_name = session.GetInputNameAllocated(0, allocator);
        auto output_name = session.GetOutputNameAllocated(0, allocator);

        const auto input_shape = session.GetInputTypeInfo(0)
                                     .GetTensorTypeAndShapeInfo()
                                     .GetShape();

        std::vector<float> input_data(element_count(input_shape), 0.0F);
        const auto memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_data.data(),
            input_data.size(),
            input_shape.data(),
            input_shape.size());

        const char* input_names[] = {input_name.get()};
        const char* output_names[] = {output_name.get()};
        auto outputs = session.Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            1);

        const auto output_shape = outputs.front()
                                      .GetTensorTypeAndShapeInfo()
                                      .GetShape();

        std::cout << "Input name:   " << input_name.get() << '\n';
        std::cout << "Input shape:  ";
        print_shape(input_shape);
        std::cout << '\n';
        std::cout << "Output name:  " << output_name.get() << '\n';
        std::cout << "Output shape: ";
        print_shape(output_shape);
        std::cout << '\n';
        std::cout << "ONNX Runtime C++ inference: OK\n";
    } catch (const Ort::Exception& error) {
        std::cerr << "ONNX Runtime error while loading or running the model:\n";
        std::cerr << error.what() << '\n';
        return 1;
    }

    return 0;
}
