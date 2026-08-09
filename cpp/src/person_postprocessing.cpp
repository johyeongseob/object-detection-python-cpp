#include "person_postprocessing.hpp"

#include <opencv2/dnn/dnn.hpp>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

std::vector<int> apply_nms(
    const std::vector<cv::Rect2d>& boxes,
    const std::vector<float>& confidences,
    float confidence_threshold,
    float nms_iou_threshold,
    int max_detections) {
    if (boxes.size() != confidences.size()) {
        throw std::invalid_argument(
            "The number of boxes and confidence scores must match.");
    }
    if (max_detections < 0) {
        throw std::invalid_argument("max_detections must not be negative.");
    }
    if (max_detections == 0 || boxes.empty()) {
        return {};
    }

    std::vector<int> retained_indices;
    cv::dnn::NMSBoxes(
        boxes,
        confidences,
        confidence_threshold,
        nms_iou_threshold,
        retained_indices,
        1.0F,
        0);

    if (retained_indices.size() >
        static_cast<std::size_t>(max_detections)) {
        retained_indices.resize(static_cast<std::size_t>(max_detections));
    }
    return retained_indices;
}
