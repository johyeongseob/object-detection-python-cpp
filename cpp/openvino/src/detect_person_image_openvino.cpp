#include "openvino_person_detector.hpp"
#include "person_visualization.hpp"

#include <opencv2/imgcodecs.hpp>

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

constexpr int kInputSize = 640;
constexpr float kConfidenceThreshold = 0.25F;
constexpr float kNmsIouThreshold = 0.70F;
constexpr int kMaxDetections = 100;
constexpr const char* kDevice = "GPU";

}  // namespace

int main(int argc, char* argv[]) {
    const std::filesystem::path image_path =
        argc > 1
            ? argv[1]
            : "datasets/coco/val2017/000000000139.jpg";
    const std::filesystem::path output_path =
        argc > 2
            ? argv[2]
            : "outputs/yolo11n/images/openvino_gpu_139_person.jpg";
    const std::filesystem::path model_path =
        argc > 3
            ? argv[3]
            : "models/yolo11n/openvino/yolo11n.xml";

    try {
        cv::Mat image = cv::imread(image_path.string());
        if (image.empty()) {
            std::cerr << "Error: failed to read image: "
                      << image_path.string() << '\n';
            return 1;
        }

        OpenVinoPersonDetector detector{
            model_path.string(),
            kDevice,
            kInputSize,
            kConfidenceThreshold,
            kNmsIouThreshold,
            kMaxDetections};
        const OpenVinoDetectionResult result = detector.detect(image);

        for (const OpenVinoDetection& detection : result.detections) {
            draw_person_detection(
                image, detection.box, detection.confidence);
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
        std::cout << "Runtime: OpenVINO C++\n";
        std::cout << "Device: " << kDevice << '\n';
        std::cout << "Input size: " << kInputSize << '\n';
        std::cout << "Confidence threshold: " << kConfidenceThreshold << '\n';
        std::cout << "NMS IoU threshold: " << kNmsIouThreshold << '\n';
        std::cout << "Detections: " << result.detections.size() << '\n';
        std::cout << "Model inference: " << std::fixed << std::setprecision(2)
                  << result.inference_ms << " ms\n";
        for (std::size_t index = 0; index < result.detections.size(); ++index) {
            const OpenVinoDetection& detection = result.detections[index];
            std::cout << "  " << index + 1 << ". person confidence="
                      << std::fixed << std::setprecision(4)
                      << detection.confidence << " bbox=["
                      << detection.box.x << ", " << detection.box.y << ", "
                      << detection.box.width << ", "
                      << detection.box.height << "]\n";
        }
        std::cout << "Saved: " << output_path.string() << '\n';
    } catch (const cv::Exception& error) {
        std::cerr << "OpenCV error: " << error.what() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
