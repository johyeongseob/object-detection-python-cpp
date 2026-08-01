#include "person_visualization.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>

namespace {

const cv::Scalar kBoxColor{255, 0, 0};
constexpr int kBoxThickness = 2;
constexpr int kFont = cv::FONT_HERSHEY_SIMPLEX;
constexpr double kFontScale = 0.6;
constexpr int kFontThickness = 1;
constexpr int kLabelPadding = 4;

}  // namespace

void draw_person_detection(
    cv::Mat& image,
    const cv::Rect2d& box,
    float confidence) {
    const int x1 = static_cast<int>(std::round(box.x));
    const int y1 = static_cast<int>(std::round(box.y));
    const int x2 = static_cast<int>(std::round(box.x + box.width));
    const int y2 = static_cast<int>(std::round(box.y + box.height));
    cv::rectangle(
        image,
        cv::Point{x1, y1},
        cv::Point{x2, y2},
        kBoxColor,
        kBoxThickness);

    std::ostringstream label_stream;
    label_stream << "person " << std::fixed << std::setprecision(2)
                 << confidence;
    const std::string label = label_stream.str();

    int baseline = 0;
    const cv::Size text_size = cv::getTextSize(
        label,
        kFont,
        kFontScale,
        kFontThickness,
        &baseline);
    const int label_height = text_size.height + 2 * kLabelPadding;
    const int label_top = std::max(y1, label_height);
    const cv::Rect background{
        x1,
        label_top - label_height,
        text_size.width + 2 * kLabelPadding,
        label_height};
    cv::rectangle(image, background, kBoxColor, cv::FILLED);
    cv::putText(
        image,
        label,
        cv::Point{x1 + kLabelPadding, label_top - kLabelPadding},
        kFont,
        kFontScale,
        cv::Scalar{255, 255, 255},
        kFontThickness,
        cv::LINE_AA);
}
