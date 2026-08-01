#include "person_detector.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kPersonClassId = 0;
constexpr int kCocoClassCount = 80;
constexpr int kBoxValueCount = 4;
constexpr int kFeatureCount = kBoxValueCount + kCocoClassCount;

struct LetterboxResult {
    cv::Mat image;
    float scale;
    int left_padding;
    int top_padding;
};

LetterboxResult letterbox(const cv::Mat& image, int input_size) {
    const float scale = std::min(
        static_cast<float>(input_size) / static_cast<float>(image.cols),
        static_cast<float>(input_size) / static_cast<float>(image.rows));

    const int resized_width =
        static_cast<int>(std::round(static_cast<float>(image.cols) * scale));
    const int resized_height =
        static_cast<int>(std::round(static_cast<float>(image.rows) * scale));

    cv::Mat resized;
    cv::resize(
        image,
        resized,
        cv::Size(resized_width, resized_height),
        0.0,
        0.0,
        cv::INTER_LINEAR);

    const float horizontal_padding =
        static_cast<float>(input_size - resized_width) / 2.0F;
    const float vertical_padding =
        static_cast<float>(input_size - resized_height) / 2.0F;
    const int left = static_cast<int>(std::round(horizontal_padding - 0.1F));
    const int right = static_cast<int>(std::round(horizontal_padding + 0.1F));
    const int top = static_cast<int>(std::round(vertical_padding - 0.1F));
    const int bottom = static_cast<int>(std::round(vertical_padding + 0.1F));

    cv::Mat padded;
    cv::copyMakeBorder(
        resized,
        padded,
        top,
        bottom,
        left,
        right,
        cv::BORDER_CONSTANT,
        cv::Scalar(114, 114, 114));

    return {padded, scale, left, top};
}

cv::Rect2d restore_box(
    float center_x,
    float center_y,
    float width,
    float height,
    const LetterboxResult& letterboxed,
    const cv::Size& original_size) {
    float left = (center_x - width / 2.0F - letterboxed.left_padding) /
                 letterboxed.scale;
    float top = (center_y - height / 2.0F - letterboxed.top_padding) /
                letterboxed.scale;
    float right = (center_x + width / 2.0F - letterboxed.left_padding) /
                  letterboxed.scale;
    float bottom = (center_y + height / 2.0F - letterboxed.top_padding) /
                   letterboxed.scale;

    left = std::clamp(left, 0.0F, static_cast<float>(original_size.width));
    top = std::clamp(top, 0.0F, static_cast<float>(original_size.height));
    right = std::clamp(right, 0.0F, static_cast<float>(original_size.width));
    bottom = std::clamp(bottom, 0.0F, static_cast<float>(original_size.height));

    return {
        static_cast<double>(left),
        static_cast<double>(top),
        static_cast<double>(std::max(0.0F, right - left)),
        static_cast<double>(std::max(0.0F, bottom - top))};
}

}  // namespace

class PersonDetector::Implementation {
public:
    Implementation(
        const std::string& model_path,
        int input_size,
        float confidence_threshold,
        float nms_iou_threshold,
        int max_detections)
        : environment_{ORT_LOGGING_LEVEL_WARNING, "person_detector"},
          session_options_{},
          session_{nullptr},
          input_size_{input_size},
          confidence_threshold_{confidence_threshold},
          nms_iou_threshold_{nms_iou_threshold},
          max_detections_{max_detections} {
        session_options_.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_ = Ort::Session{
            environment_, model_path.c_str(), session_options_};

        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name = session_.GetInputNameAllocated(0, allocator);
        auto output_name = session_.GetOutputNameAllocated(0, allocator);
        input_name_ = input_name.get();
        output_name_ = output_name.get();

        const auto input_shape = session_.GetInputTypeInfo(0)
                                     .GetTensorTypeAndShapeInfo()
                                     .GetShape();
        if (input_shape != std::vector<std::int64_t>{
                               1, 3, input_size_, input_size_}) {
            throw std::runtime_error(
                "The model input shape does not match the configured size.");
        }
    }

    DetectionResult detect(const cv::Mat& image) {
        if (image.empty()) {
            throw std::invalid_argument("The input image is empty.");
        }

        const LetterboxResult letterboxed = letterbox(image, input_size_);
        cv::Mat blob = cv::dnn::blobFromImage(
            letterboxed.image,
            1.0 / 255.0,
            cv::Size(input_size_, input_size_),
            cv::Scalar(),
            true,
            false,
            CV_32F);

        const std::vector<std::int64_t> input_shape{
            1, 3, input_size_, input_size_};
        const auto memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            blob.ptr<float>(),
            blob.total(),
            input_shape.data(),
            input_shape.size());

        const char* input_names[] = {input_name_.c_str()};
        const char* output_names[] = {output_name_.c_str()};
        const auto inference_start = std::chrono::steady_clock::now();
        auto outputs = session_.Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            1);
        const auto inference_end = std::chrono::steady_clock::now();
        const double inference_ms =
            std::chrono::duration<double, std::milli>(
                inference_end - inference_start)
                .count();

        const auto output_shape = outputs.front()
                                      .GetTensorTypeAndShapeInfo()
                                      .GetShape();
        if (output_shape.size() != 3 || output_shape[0] != 1 ||
            output_shape[1] != kFeatureCount) {
            throw std::runtime_error("Unexpected YOLO11 output shape.");
        }

        const std::size_t candidate_count =
            static_cast<std::size_t>(output_shape[2]);
        const float* output = outputs.front().GetTensorData<float>();

        std::vector<cv::Rect2d> boxes;
        std::vector<float> confidences;
        boxes.reserve(candidate_count);
        confidences.reserve(candidate_count);

        for (std::size_t candidate = 0; candidate < candidate_count;
             ++candidate) {
            int predicted_class = 0;
            float confidence = output[kBoxValueCount * candidate_count + candidate];
            for (int class_id = 1; class_id < kCocoClassCount; ++class_id) {
                const float class_score =
                    output[(kBoxValueCount + class_id) * candidate_count +
                           candidate];
                if (class_score > confidence) {
                    confidence = class_score;
                    predicted_class = class_id;
                }
            }
            if (predicted_class != kPersonClassId) {
                continue;
            }
            if (confidence < confidence_threshold_) {
                continue;
            }

            const float center_x = output[candidate];
            const float center_y = output[candidate_count + candidate];
            const float width = output[2 * candidate_count + candidate];
            const float height = output[3 * candidate_count + candidate];
            cv::Rect2d box = restore_box(
                center_x,
                center_y,
                width,
                height,
                letterboxed,
                image.size());
            if (box.area() <= 0) {
                continue;
            }

            boxes.push_back(box);
            confidences.push_back(confidence);
        }

        std::vector<int> retained_indices;
        cv::dnn::NMSBoxes(
            boxes,
            confidences,
            confidence_threshold_,
            nms_iou_threshold_,
            retained_indices,
            1.0F,
            0);

        if (retained_indices.size() >
            static_cast<std::size_t>(max_detections_)) {
            retained_indices.resize(
                static_cast<std::size_t>(max_detections_));
        }

        std::vector<Detection> detections;
        detections.reserve(retained_indices.size());
        for (const int index : retained_indices) {
            detections.push_back(
                {kPersonClassId, confidences[index], boxes[index]});
        }
        return {std::move(detections), inference_ms};
    }

private:
    Ort::Env environment_;
    Ort::SessionOptions session_options_;
    Ort::Session session_;
    std::string input_name_;
    std::string output_name_;
    int input_size_;
    float confidence_threshold_;
    float nms_iou_threshold_;
    int max_detections_;
};

PersonDetector::PersonDetector(
    const std::string& model_path,
    int input_size,
    float confidence_threshold,
    float nms_iou_threshold,
    int max_detections)
    : implementation_{std::make_unique<Implementation>(
          model_path,
          input_size,
          confidence_threshold,
          nms_iou_threshold,
          max_detections)} {}

PersonDetector::~PersonDetector() = default;
PersonDetector::PersonDetector(PersonDetector&&) noexcept = default;
PersonDetector& PersonDetector::operator=(PersonDetector&&) noexcept = default;

DetectionResult PersonDetector::detect(const cv::Mat& image) {
    return implementation_->detect(image);
}
