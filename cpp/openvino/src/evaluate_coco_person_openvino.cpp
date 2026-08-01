#include "openvino_person_detector.hpp"

#include <openvino/openvino.hpp>
#include <opencv2/core/version.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kInputSize = 640;
constexpr float kConfidenceThreshold = 0.001F;
constexpr float kNmsIouThreshold = 0.70F;
constexpr int kMaxDetections = 100;
constexpr int kCocoPersonCategoryId = 1;
constexpr std::size_t kDefaultWarmupIterations = 10;
constexpr std::size_t kProgressInterval = 100;
constexpr const char* kDevice = "GPU";

struct Options {
    std::size_t limit = 0;
    std::size_t warmup_iterations = kDefaultWarmupIterations;
};

struct Prediction {
    int image_id;
    cv::Rect2d box;
    float score;
};

Options parse_options(int argc, char* argv[]) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--limit" && index + 1 < argc) {
            options.limit = std::stoul(argv[++index]);
        } else if (argument == "--warmup" && index + 1 < argc) {
            options.warmup_iterations = std::stoul(argv[++index]);
        } else {
            throw std::invalid_argument(
                "Usage: evaluate_coco_person_openvino "
                "[--limit N] [--warmup N]");
        }
    }
    return options;
}

std::vector<std::filesystem::path> collect_images(
    const std::filesystem::path& image_directory,
    std::size_t limit) {
    if (!std::filesystem::is_directory(image_directory)) {
        throw std::runtime_error(
            "Image directory not found: " + image_directory.string());
    }

    std::vector<std::filesystem::path> image_paths;
    for (const auto& entry :
         std::filesystem::directory_iterator(image_directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
            image_paths.push_back(entry.path());
        }
    }
    std::sort(image_paths.begin(), image_paths.end());
    if (limit > 0 && limit < image_paths.size()) {
        image_paths.resize(limit);
    }
    if (image_paths.empty()) {
        throw std::runtime_error("No JPEG images found.");
    }
    return image_paths;
}

double percentile(std::vector<double> values, double quantile) {
    if (values.empty()) {
        throw std::invalid_argument("Cannot calculate an empty percentile.");
    }
    std::sort(values.begin(), values.end());
    const double position =
        quantile * static_cast<double>(values.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(position);
    const std::size_t upper = std::min(lower + 1, values.size() - 1);
    const double fraction = position - static_cast<double>(lower);
    return values[lower] + (values[upper] - values[lower]) * fraction;
}

double mean(const std::vector<double>& values) {
    double total = 0.0;
    for (const double value : values) {
        total += value;
    }
    return total / static_cast<double>(values.size());
}

void write_predictions(
    const std::filesystem::path& path,
    const std::vector<Prediction>& predictions) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open prediction file.");
    }

    file << "[\n" << std::setprecision(10);
    for (std::size_t index = 0; index < predictions.size(); ++index) {
        const Prediction& prediction = predictions[index];
        file << "  {\"image_id\":" << prediction.image_id
             << ",\"category_id\":" << kCocoPersonCategoryId
             << ",\"bbox\":[" << prediction.box.x << ','
             << prediction.box.y << ',' << prediction.box.width << ','
             << prediction.box.height << "],\"score\":"
             << prediction.score << '}';
        file << (index + 1 < predictions.size() ? ",\n" : "\n");
    }
    file << "]\n";
}

void write_runtime_summary(
    const std::filesystem::path& path,
    std::size_t image_count,
    std::size_t prediction_count,
    std::size_t warmup_iterations,
    double wall_time_seconds,
    const std::vector<double>& inference_times) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open runtime summary file.");
    }

    const ov::Version version = ov::get_openvino_version();
    file << std::setprecision(10)
         << "{\n"
         << "  \"model\": {\n"
         << "    \"name\": \"yolo11n\",\n"
         << "    \"path\": "
            "\"models/yolo11n/openvino/yolo11n.xml\",\n"
         << "    \"input_size\": " << kInputSize << ",\n"
         << "    \"runtime\": \"OpenVINO C++\"\n"
         << "  },\n"
         << "  \"dataset\": {\n"
         << "    \"name\": \"COCO val2017\",\n"
         << "    \"images\": " << image_count << ",\n"
         << "    \"category\": \"person\",\n"
         << "    \"category_id\": " << kCocoPersonCategoryId << "\n"
         << "  },\n"
         << "  \"evaluation\": {\n"
         << "    \"confidence_threshold\": 0.001,\n"
         << "    \"nms_iou_threshold\": 0.7,\n"
         << "    \"max_detections\": " << kMaxDetections << ",\n"
         << "    \"predictions\": " << prediction_count << "\n"
         << "  },\n"
         << "  \"performance\": {\n"
         << "    \"device\": \"gpu\",\n"
         << "    \"warmup_iterations\": " << warmup_iterations << ",\n"
         << "    \"wall_time_seconds\": " << wall_time_seconds << ",\n"
         << "    \"throughput_images_per_second\": "
         << static_cast<double>(image_count) / wall_time_seconds << ",\n"
         << "    \"mean_inference_ms\": " << mean(inference_times) << ",\n"
         << "    \"p50_inference_ms\": "
         << percentile(inference_times, 0.50) << ",\n"
         << "    \"p95_inference_ms\": "
         << percentile(inference_times, 0.95) << "\n"
         << "  },\n"
         << "  \"environment\": {\n"
         << "    \"openvino\": \"" << version.buildNumber << "\",\n"
         << "    \"openvino_device\": \"GPU\",\n"
         << "    \"opencv_cpp\": \"" << CV_VERSION << "\"\n"
         << "  }\n"
         << "}\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    const std::filesystem::path image_directory =
        "datasets/coco/val2017";
    const std::filesystem::path model_path =
        "models/yolo11n/openvino/yolo11n.xml";

    try {
        const Options options = parse_options(argc, argv);
        const auto image_paths = collect_images(image_directory, options.limit);
        const std::string suffix =
            options.limit == 0
                ? ""
                : "_" + std::to_string(image_paths.size()) + "_images";
        const std::filesystem::path prediction_path =
            "outputs/yolo11n/"
            "openvino_gpu_coco_val2017_person_predictions" + suffix +
            ".json";
        const std::filesystem::path runtime_path =
            "results/yolo11n/openvino_gpu_coco_person_runtime" + suffix +
            ".json";

        std::cout << "\nEvaluation settings\n"
                  << "  Images:               " << image_paths.size() << '\n'
                  << "  Class:                person\n"
                  << "  Confidence threshold: " << kConfidenceThreshold << '\n'
                  << "  NMS IoU threshold:    " << kNmsIouThreshold << '\n'
                  << "  Input size:           " << kInputSize << '\n'
                  << "  Device:               gpu\n"
                  << "  Runtime:              OpenVINO C++\n"
                  << "  Warm-up iterations:   "
                  << options.warmup_iterations << std::endl;

        OpenVinoPersonDetector detector{
            model_path.string(),
            kDevice,
            kInputSize,
            kConfidenceThreshold,
            kNmsIouThreshold,
            kMaxDetections};

        const cv::Mat warmup_image = cv::imread(image_paths.front().string());
        if (warmup_image.empty()) {
            throw std::runtime_error("Failed to read warm-up image.");
        }
        for (std::size_t iteration = 0;
             iteration < options.warmup_iterations;
             ++iteration) {
            static_cast<void>(detector.detect(warmup_image));
        }

        std::vector<Prediction> predictions;
        std::vector<double> inference_times;
        inference_times.reserve(image_paths.size());
        const auto wall_start = std::chrono::steady_clock::now();

        for (std::size_t index = 0; index < image_paths.size(); ++index) {
            const std::filesystem::path& image_path = image_paths[index];
            const int image_id = std::stoi(image_path.stem().string());
            const cv::Mat image = cv::imread(image_path.string());
            if (image.empty()) {
                throw std::runtime_error(
                    "Failed to read image: " + image_path.string());
            }

            OpenVinoDetectionResult result = detector.detect(image);
            inference_times.push_back(result.inference_ms);
            for (const OpenVinoDetection& detection : result.detections) {
                predictions.push_back(
                    {image_id, detection.box, detection.confidence});
            }

            const std::size_t processed = index + 1;
            if (processed % kProgressInterval == 0 ||
                processed == image_paths.size()) {
                const double elapsed =
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - wall_start)
                        .count();
                std::cout << "  Processed " << std::setw(4) << processed << '/'
                          << image_paths.size() << " images (" << std::fixed
                          << std::setprecision(2)
                          << static_cast<double>(processed) / elapsed
                          << " images/s)" << std::endl;
            }
        }

        const double wall_time_seconds =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - wall_start)
                .count();

        write_predictions(prediction_path, predictions);
        write_runtime_summary(
            runtime_path,
            image_paths.size(),
            predictions.size(),
            options.warmup_iterations,
            wall_time_seconds,
            inference_times);

        std::cout << "\nPredictions: " << predictions.size() << '\n'
                  << "Prediction file: " << prediction_path.string() << '\n'
                  << "Wall time: " << std::fixed << std::setprecision(2)
                  << wall_time_seconds << " s\n"
                  << "Throughput: "
                  << static_cast<double>(image_paths.size()) /
                         wall_time_seconds
                  << " images/s\n"
                  << "Mean model inference: " << mean(inference_times)
                  << " ms\n"
                  << "P50 model inference:  "
                  << percentile(inference_times, 0.50) << " ms\n"
                  << "P95 model inference:  "
                  << percentile(inference_times, 0.95) << " ms\n"
                  << "Runtime summary: " << runtime_path.string() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
