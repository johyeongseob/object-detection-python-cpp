#include "person_detector.hpp"
#include "person_visualization.hpp"

#include <httplib.h>
#include <opencv2/imgcodecs.hpp>

#include <cstddef>
#include <exception>
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

const char* kIndexHtml = R"HTML(<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>YOLO11n Person Detection</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; }
    body { margin: 0; background: #111827; color: #f9fafb; }
    main { width: min(900px, calc(100% - 32px)); margin: 48px auto; }
    .card { background: #1f2937; border: 1px solid #374151; border-radius: 16px; padding: 24px; }
    h1 { margin-top: 0; }
    p { color: #d1d5db; }
    .controls { display: flex; gap: 12px; flex-wrap: wrap; margin: 20px 0; }
    input { flex: 1; min-width: 240px; padding: 10px; background: #111827; border: 1px solid #4b5563; border-radius: 8px; }
    button { padding: 10px 18px; border: 0; border-radius: 8px; background: #2563eb; color: white; font-weight: 700; cursor: pointer; }
    button:disabled { opacity: 0.55; cursor: wait; }
    #status { min-height: 24px; }
    #result { display: none; max-width: 100%; margin-top: 16px; border-radius: 10px; }
  </style>
</head>
<body>
  <main>
    <div class="card">
      <h1>YOLO11n Person Detection</h1>
      <p>Select an image to run the C++ ONNX Runtime detector.</p>
      <div class="controls">
        <input id="file" type="file" accept="image/*">
        <button id="detect">Detect people</button>
      </div>
      <div id="status">Ready</div>
      <img id="result" alt="Person detection result">
    </div>
  </main>
  <script>
    const fileInput = document.getElementById('file');
    const button = document.getElementById('detect');
    const status = document.getElementById('status');
    const result = document.getElementById('result');
    let resultUrl = null;

    button.addEventListener('click', async () => {
      const file = fileInput.files[0];
      if (!file) {
        status.textContent = 'Choose an image first.';
        return;
      }

      button.disabled = true;
      result.style.display = 'none';
      status.textContent = 'Running detection...';
      try {
        const response = await fetch('/detect', {
          method: 'POST',
          headers: { 'Content-Type': file.type || 'application/octet-stream' },
          body: file
        });
        if (!response.ok) {
          throw new Error(await response.text());
        }

        const blob = await response.blob();
        if (resultUrl) URL.revokeObjectURL(resultUrl);
        resultUrl = URL.createObjectURL(blob);
        result.src = resultUrl;
        result.style.display = 'block';
        const count = response.headers.get('X-Detection-Count');
        const latency = response.headers.get('X-Inference-Ms');
        status.textContent = `${count} person(s) detected | ${latency} ms model inference`;
      } catch (error) {
        status.textContent = `Error: ${error.message}`;
      } finally {
        button.disabled = false;
      }
    });
  </script>
</body>
</html>)HTML";

void set_error(
    httplib::Response& response,
    int status,
    const std::string& message) {
    response.status = status;
    response.set_content(message, "text/plain; charset=utf-8");
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
        PersonDetector detector{
            model_path,
            kInputSize,
            kConfidenceThreshold,
            kNmsIouThreshold,
            kMaxDetections};
        std::mutex detector_mutex;

        httplib::Server server;
        server.set_payload_max_length(kMaximumUploadBytes);
        server.Get("/", [](const httplib::Request&, httplib::Response& response) {
            response.set_content(kIndexHtml, "text/html; charset=utf-8");
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
