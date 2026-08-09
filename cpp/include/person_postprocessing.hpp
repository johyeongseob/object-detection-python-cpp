#pragma once

#include <opencv2/core/types.hpp>

#include <vector>

std::vector<int> apply_nms(
    const std::vector<cv::Rect2d>& boxes,
    const std::vector<float>& confidences,
    float confidence_threshold,
    float nms_iou_threshold,
    int max_detections);
