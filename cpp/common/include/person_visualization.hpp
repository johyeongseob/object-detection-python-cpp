#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

void draw_person_detection(
    cv::Mat& image,
    const cv::Rect2d& box,
    float confidence);
