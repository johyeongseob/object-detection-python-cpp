#include "openvino_person_detector.hpp"

#include <openvino/openvino.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
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

class OpenVinoPersonDetector::Implementation {
public:
    Implementation(
        const std::string& model_path,
        const std::string& device,
        int input_size,
        float confidence_threshold,
        float nms_iou_threshold,
        int max_detections)
        : input_size_{input_size},
          confidence_threshold_{confidence_threshold},
          nms_iou_threshold_{nms_iou_threshold},
          max_detections_{max_detections} {
        const std::shared_ptr<ov::Model> model = core_.read_model(model_path);

        const ov::Shape expected_input_shape{
            1,
            3,
            static_cast<std::size_t>(input_size_),
            static_cast<std::size_t>(input_size_)};
        if (model->input().get_shape() != expected_input_shape) {
            throw std::runtime_error(
                "The model input shape does not match the configured size.");
        }
        if (model->input().get_element_type() != ov::element::f32) {
            throw std::runtime_error("The model input type must be FP32.");
        }

        compiled_model_ = core_.compile_model(model, device);
        infer_request_ = compiled_model_.create_infer_request();
    }

    OpenVinoDetectionResult detect(const cv::Mat& image) {
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

        ov::Tensor input_tensor{
            ov::element::f32,
            ov::Shape{
                1,
                3,
                static_cast<std::size_t>(input_size_),
                static_cast<std::size_t>(input_size_)}};
        std::memcpy(
            input_tensor.data<float>(),
            blob.ptr<float>(),
            blob.total() * sizeof(float));
        infer_request_.set_input_tensor(input_tensor);

        const auto inference_start = std::chrono::steady_clock::now();
        infer_request_.infer();
        const auto inference_end = std::chrono::steady_clock::now();
        const double inference_ms =
            std::chrono::duration<double, std::milli>(
                inference_end - inference_start)
                .count();

        const ov::Tensor output_tensor = infer_request_.get_output_tensor();
        const ov::Shape& output_shape = output_tensor.get_shape();
        if (output_tensor.get_element_type() != ov::element::f32 ||
            output_shape.size() != 3 || output_shape[0] != 1 ||
            output_shape[1] != kFeatureCount) {
            throw std::runtime_error("Unexpected YOLO11 output tensor.");
        }

        const std::size_t candidate_count = output_shape[2];
        const float* output = output_tensor.data<const float>();

        std::vector<cv::Rect2d> boxes;
        std::vector<float> confidences;
        boxes.reserve(candidate_count);
        confidences.reserve(candidate_count);

        for (std::size_t candidate = 0; candidate < candidate_count;
             ++candidate) {
            int predicted_class = 0;
            float confidence =
                output[kBoxValueCount * candidate_count + candidate];
            for (int class_id = 1; class_id < kCocoClassCount; ++class_id) {
                const float class_score =
                    output[(kBoxValueCount + class_id) * candidate_count +
                           candidate];
                if (class_score > confidence) {
                    confidence = class_score;
                    predicted_class = class_id;
                }
            }
            if (predicted_class != kPersonClassId ||
                confidence < confidence_threshold_) {
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

        std::vector<OpenVinoDetection> detections;
        detections.reserve(retained_indices.size());
        for (const int index : retained_indices) {
            detections.push_back(
                {kPersonClassId, confidences[index], boxes[index]});
        }

        return {std::move(detections), inference_ms};
    }

private:
    ov::Core core_;
    ov::CompiledModel compiled_model_;
    ov::InferRequest infer_request_;
    int input_size_;
    float confidence_threshold_;
    float nms_iou_threshold_;
    int max_detections_;
};

OpenVinoPersonDetector::OpenVinoPersonDetector(
    const std::string& model_path,
    const std::string& device,
    int input_size,
    float confidence_threshold,
    float nms_iou_threshold,
    int max_detections)
    : implementation_{std::make_unique<Implementation>(
          model_path,
          device,
          input_size,
          confidence_threshold,
          nms_iou_threshold,
          max_detections)} {}

OpenVinoPersonDetector::~OpenVinoPersonDetector() = default;
OpenVinoPersonDetector::OpenVinoPersonDetector(
    OpenVinoPersonDetector&&) noexcept = default;
OpenVinoPersonDetector& OpenVinoPersonDetector::operator=(
    OpenVinoPersonDetector&&) noexcept = default;

OpenVinoDetectionResult OpenVinoPersonDetector::detect(const cv::Mat& image) {
    return implementation_->detect(image);
}
