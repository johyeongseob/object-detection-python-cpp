#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include <memory>
#include <string>
#include <vector>

struct Detection {
    int class_id;
    float confidence;
    cv::Rect2d box;
};

struct DetectionResult {
    std::vector<Detection> detections;
    double inference_ms;
};

class PersonDetector {
public:
    PersonDetector(
        const std::string& model_path,
        int input_size,
        float confidence_threshold,
        float nms_iou_threshold,
        int max_detections);

    ~PersonDetector();

    PersonDetector(const PersonDetector&) = delete;
    PersonDetector& operator=(const PersonDetector&) = delete;
    PersonDetector(PersonDetector&&) noexcept;
    PersonDetector& operator=(PersonDetector&&) noexcept;

    DetectionResult detect(const cv::Mat& image);

private:
    class Implementation;
    std::unique_ptr<Implementation> implementation_;
};
