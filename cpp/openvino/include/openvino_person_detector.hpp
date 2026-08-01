#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include <memory>
#include <string>
#include <vector>

struct OpenVinoDetection {
    int class_id;
    float confidence;
    cv::Rect2d box;
};

struct OpenVinoDetectionResult {
    std::vector<OpenVinoDetection> detections;
    double inference_ms;
};

class OpenVinoPersonDetector {
public:
    OpenVinoPersonDetector(
        const std::string& model_path,
        const std::string& device,
        int input_size,
        float confidence_threshold,
        float nms_iou_threshold,
        int max_detections);

    ~OpenVinoPersonDetector();

    OpenVinoPersonDetector(const OpenVinoPersonDetector&) = delete;
    OpenVinoPersonDetector& operator=(const OpenVinoPersonDetector&) = delete;
    OpenVinoPersonDetector(OpenVinoPersonDetector&&) noexcept;
    OpenVinoPersonDetector& operator=(OpenVinoPersonDetector&&) noexcept;

    OpenVinoDetectionResult detect(const cv::Mat& image);

private:
    class Implementation;
    std::unique_ptr<Implementation> implementation_;
};
