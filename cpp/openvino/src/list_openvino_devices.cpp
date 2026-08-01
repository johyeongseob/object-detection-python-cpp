#include <openvino/openvino.hpp>

#include <exception>
#include <iostream>
#include <string>
#include <vector>

int main() {
    try {
        ov::Core core;
        const std::vector<std::string> devices = core.get_available_devices();

        std::cout << "OpenVINO version: " << ov::get_openvino_version().description
                  << '\n';
        std::cout << "Available OpenVINO devices (" << devices.size() << "):\n";

        if (devices.empty()) {
            std::cerr << "  No OpenVINO devices were found.\n";
            return 1;
        }

        bool gpu_found = false;
        for (const std::string& device : devices) {
            std::cout << "  - " << device;

            try {
                const std::string full_name =
                    core.get_property(device, ov::device::full_name);
                std::cout << ": " << full_name;
            } catch (const std::exception&) {
                // Some plugins may not expose a full device name.
            }

            std::cout << '\n';

            if (device == "GPU" || device.rfind("GPU.", 0) == 0) {
                gpu_found = true;
            }
        }

        if (!gpu_found) {
            std::cerr << "OpenVINO GPU device: NOT FOUND\n";
            return 2;
        }

        std::cout << "OpenVINO GPU device: OK\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "OpenVINO error: " << error.what() << '\n';
        return 1;
    }
}
