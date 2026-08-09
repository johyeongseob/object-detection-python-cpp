#include "person_detector.hpp"
#include "person_visualization.hpp"

#include <httplib.h>
#include <opencv2/imgcodecs.hpp>

#include <cstddef>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kInputSize = 640;
constexpr float kConfidenceThreshold = 0.25F;
constexpr float kNmsIouThreshold = 0.70F;
constexpr int kMaxDetections = 100;
constexpr int kDefaultPort = 8080;
constexpr std::size_t kMaximumUploadBytes = 20U * 1024U * 1024U;

void set_error(
    httplib::Response& response,
    int status,
    const std::string& message) {
    response.status = status;
    response.set_content(message, "text/plain; charset=utf-8");
}

void serve_file(
    httplib::Response& response,
    const std::filesystem::path& path,
    const std::string& content_type) {
    if (!std::filesystem::is_regular_file(path)) {
        set_error(response, 500, "Web asset not found: " + path.string());
        return;
    }
    response.set_file_content(path.string(), content_type);
}

int parse_port(const char* value) {
    const int port = std::stoi(value);
    if (port < 1 || port > 65535) {
        throw std::invalid_argument("Port must be between 1 and 65535.");
    }
    return port;
}

}  // namespace

int main(int argc, char* argv[]) {
    const std::string model_path =
        argc > 1 ? argv[1] : "models/yolo11n/yolo11n.onnx";

    try {
        const int port = argc > 2 ? parse_port(argv[2]) : kDefaultPort;
        const std::filesystem::path web_root = argc > 3 ? argv[3] : "web";
        const std::filesystem::path index_path = web_root / "index.html";
        const std::filesystem::path style_path = web_root / "style.css";
        const std::filesystem::path app_path = web_root / "app.js";

        PersonDetector detector{
            model_path,
            kInputSize,
            kConfidenceThreshold,
            kNmsIouThreshold,
            kMaxDetections};
        std::mutex detector_mutex;

        httplib::Server server;
        server.set_payload_max_length(kMaximumUploadBytes);
        server.Get(
            "/",
            [&](const httplib::Request&, httplib::Response& response) {
                serve_file(response, index_path, "text/html; charset=utf-8");
            });
        server.Get(
            "/style.css",
            [&](const httplib::Request&, httplib::Response& response) {
                serve_file(response, style_path, "text/css; charset=utf-8");
            });
        server.Get(
            "/app.js",
            [&](const httplib::Request&, httplib::Response& response) {
                serve_file(
                    response,
                    app_path,
                    "application/javascript; charset=utf-8");
            });
        server.Get(
            "/health",
            [](const httplib::Request&, httplib::Response& response) {
                response.set_content(
                    R"({"status":"ok","runtime":"ONNX Runtime C++"})",
                    "application/json");
            });
        server.Post(
            "/detect",
            [&](const httplib::Request& request, httplib::Response& response) {
                if (request.body.empty()) {
                    set_error(response, 400, "The request body is empty.");
                    return;
                }

                try {
                    const std::vector<unsigned char> encoded_input(
                        request.body.begin(), request.body.end());
                    cv::Mat image = cv::imdecode(encoded_input, cv::IMREAD_COLOR);
                    if (image.empty()) {
                        set_error(response, 400, "The uploaded file is not a valid image.");
                        return;
                    }

                    DetectionResult result;
                    {
                        const std::lock_guard<std::mutex> lock{detector_mutex};
                        result = detector.detect(image);
                    }
                    for (const Detection& detection : result.detections) {
                        draw_person_detection(
                            image, detection.box, detection.confidence);
                    }

                    std::vector<unsigned char> encoded_output;
                    if (!cv::imencode(".jpg", image, encoded_output)) {
                        throw std::runtime_error("Failed to encode the result image.");
                    }

                    std::ostringstream latency;
                    latency << std::fixed << std::setprecision(2)
                            << result.inference_ms;
                    response.set_header(
                        "X-Detection-Count",
                        std::to_string(result.detections.size()));
                    response.set_header("X-Inference-Ms", latency.str());
                    response.set_content(
                        std::string{
                            reinterpret_cast<const char*>(encoded_output.data()),
                            encoded_output.size()},
                        "image/jpeg");
                } catch (const std::exception& error) {
                    set_error(response, 500, error.what());
                }
            });

        std::cout << "Model: " << model_path << '\n';
        std::cout << "Web assets: " << web_root.string() << '\n';
        std::cout << "Server: http://localhost:" << port << '\n';
        std::cout << "Health: http://localhost:" << port << "/health\n";
        if (!server.listen("0.0.0.0", port)) {
            std::cerr << "Error: failed to listen on port " << port << '\n';
            return 1;
        }
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
