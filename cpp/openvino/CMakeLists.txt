cmake_minimum_required(VERSION 3.16)

project(yolo11n_openvino_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenVINO REQUIRED COMPONENTS Runtime)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc dnn)

add_library(person_visualization STATIC
    ../common/src/person_visualization.cpp
)

target_include_directories(person_visualization PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/../common/include"
)

target_link_libraries(person_visualization PUBLIC
    ${OpenCV_LIBS}
)

target_compile_options(person_visualization PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

add_executable(list_openvino_devices
    src/list_openvino_devices.cpp
)

target_link_libraries(list_openvino_devices PRIVATE
    openvino::runtime
)

target_compile_options(list_openvino_devices PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

add_library(openvino_person_detector STATIC
    src/openvino_person_detector.cpp
)

target_include_directories(openvino_person_detector PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(openvino_person_detector PUBLIC
    openvino::runtime
    ${OpenCV_LIBS}
)

target_compile_options(openvino_person_detector PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

add_executable(detect_person_image_openvino
    src/detect_person_image_openvino.cpp
)

target_link_libraries(detect_person_image_openvino PRIVATE
    openvino_person_detector
    person_visualization
)

target_compile_options(detect_person_image_openvino PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

add_executable(evaluate_coco_person_openvino
    src/evaluate_coco_person_openvino.cpp
)

target_link_libraries(evaluate_coco_person_openvino PRIVATE
    openvino_person_detector
)

target_compile_options(evaluate_coco_person_openvino PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)
