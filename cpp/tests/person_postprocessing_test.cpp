#include "person_postprocessing.hpp"

#include <opencv2/core/types.hpp>

#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect_indices(
    const std::string& test_name,
    const std::vector<int>& actual,
    const std::vector<int>& expected) {
    if (actual == expected) {
        return;
    }

    std::cerr << "FAILED: " << test_name << '\n';
    ++failures;
}

void test_overlapping_box_is_suppressed() {
    const std::vector<cv::Rect2d> boxes{
        {0.0, 0.0, 100.0, 100.0},
        {5.0, 5.0, 100.0, 100.0},
        {200.0, 200.0, 50.0, 50.0},
    };
    const std::vector<float> confidences{0.90F, 0.80F, 0.70F};

    expect_indices(
        "overlapping box is suppressed",
        apply_nms(boxes, confidences, 0.25F, 0.50F, 100),
        {0, 2});
}

void test_confidence_threshold_is_applied() {
    const std::vector<cv::Rect2d> boxes{
        {0.0, 0.0, 50.0, 50.0},
        {100.0, 100.0, 50.0, 50.0},
    };
    const std::vector<float> confidences{0.90F, 0.20F};

    expect_indices(
        "confidence threshold is applied",
        apply_nms(boxes, confidences, 0.25F, 0.50F, 100),
        {0});
}

void test_maximum_detection_count_is_applied() {
    const std::vector<cv::Rect2d> boxes{
        {0.0, 0.0, 50.0, 50.0},
        {100.0, 100.0, 50.0, 50.0},
    };
    const std::vector<float> confidences{0.90F, 0.80F};

    expect_indices(
        "maximum detection count is applied",
        apply_nms(boxes, confidences, 0.25F, 0.50F, 1),
        {0});
}

void test_mismatched_input_sizes_are_rejected() {
    try {
        apply_nms(
            {{0.0, 0.0, 50.0, 50.0}},
            {},
            0.25F,
            0.50F,
            100);
    } catch (const std::invalid_argument&) {
        return;
    } catch (const std::exception&) {
    }

    std::cerr << "FAILED: mismatched input sizes are rejected\n";
    ++failures;
}

}  // namespace

int main() {
    test_overlapping_box_is_suppressed();
    test_confidence_threshold_is_applied();
    test_maximum_detection_count_is_applied();
    test_mismatched_input_sizes_are_rejected();

    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }

    std::cout << "All person postprocessing tests passed\n";
    return 0;
}
