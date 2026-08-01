#include "person_detector.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {

constexpr int kInputSize = 640;
constexpr float kConfidenceThreshold = 0.25F;
constexpr float kNmsIouThreshold = 0.70F;
constexpr int kMaxDetections = 100;

void draw_detection(cv::Mat& image, const Detection& detection) {
    const cv::Scalar color{255, 0, 0};
    const cv::Rect integer_box{
        static_cast<int>(std::round(detection.box.x)),
        static_cast<int>(std::round(detection.box.y)),
        static_cast<int>(std::round(detection.box.width)),
        static_cast<int>(std::round(detection.box.height))};
    cv::rectangle(image, integer_box, color, 2);

    std::ostringstream label_stream;
    label_stream << "person " << std::fixed << std::setprecision(2)
                 << detection.confidence;
    const std::string label = label_stream.str();

    int baseline = 0;
    const cv::Size text_size = cv::getTextSize(
        label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);
    const int label_top = std::max(integer_box.y, text_size.height + 8);
    const cv::Rect background{
        integer_box.x,
        label_top - text_size.height - 8,
        text_size.width + 8,
        text_size.height + 8};
    cv::rectangle(image, background, color, cv::FILLED);
    cv::putText(
        image,
        label,
        cv::Point(integer_box.x + 4, label_top - 4),
        cv::FONT_HERSHEY_SIMPLEX,
        0.6,
        cv::Scalar{255, 255, 255},
        1,
        cv::LINE_AA);
}

}  // namespace

int main(int argc, char* argv[]) {
    const std::filesystem::path image_path =
        argc > 1
            ? argv[1]
            : "datasets/coco/val2017/000000000139.jpg";
    const std::filesystem::path output_path =
        argc > 2
            ? argv[2]
            : "outputs/yolo11n/images/cpp_139_person.jpg";
    const std::filesystem::path model_path =
        argc > 3 ? argv[3] : "models/yolo11n/yolo11n.onnx";

    try {
        cv::Mat image = cv::imread(image_path.string());
        if (image.empty()) {
            std::cerr << "Error: failed to read image: "
                      << image_path.string() << '\n';
            return 1;
        }

        PersonDetector detector{
            model_path.string(),
            kInputSize,
            kConfidenceThreshold,
            kNmsIouThreshold,
            kMaxDetections};
        const DetectionResult result = detector.detect(image);

        for (const Detection& detection : result.detections) {
            draw_detection(image, detection);
        }

        if (!output_path.parent_path().empty()) {
            std::filesystem::create_directories(output_path.parent_path());
        }
        if (!cv::imwrite(output_path.string(), image)) {
            std::cerr << "Error: failed to save image: "
                      << output_path.string() << '\n';
            return 1;
        }

        std::cout << "Image: " << image_path.string() << '\n';
        std::cout << "Model: " << model_path.string() << '\n';
        std::cout << "Input size: " << kInputSize << '\n';
        std::cout << "Confidence threshold: " << kConfidenceThreshold << '\n';
        std::cout << "NMS IoU threshold: " << kNmsIouThreshold << '\n';
        std::cout << "Detections: " << result.detections.size() << '\n';
        std::cout << "Model inference: " << std::fixed << std::setprecision(2)
                  << result.inference_ms << " ms\n";
        for (std::size_t index = 0; index < result.detections.size(); ++index) {
            const Detection& detection = result.detections[index];
            std::cout << "  " << index + 1 << ". person confidence="
                      << std::fixed << std::setprecision(4)
                      << detection.confidence << " bbox=["
                      << detection.box.x << ", " << detection.box.y << ", "
                      << detection.box.width << ", "
                      << detection.box.height << "]\n";
        }
        std::cout << "Saved: " << output_path.string() << '\n';
    } catch (const Ort::Exception& error) {
        std::cerr << "ONNX Runtime error: " << error.what() << '\n';
        return 1;
    } catch (const cv::Exception& error) {
        std::cerr << "OpenCV error: " << error.what() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
