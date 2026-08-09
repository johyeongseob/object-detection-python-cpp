cmake_minimum_required(VERSION 3.16)

project(yolo11n_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(
    ONNXRUNTIME_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-linux-x64-1.23.2"
    CACHE PATH
    "Path to the extracted ONNX Runtime C++ SDK"
)

if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "ONNX Runtime headers not found: ${ONNXRUNTIME_ROOT}")
endif()

if(NOT EXISTS "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
    message(FATAL_ERROR "ONNX Runtime library not found: ${ONNXRUNTIME_ROOT}")
endif()

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include"
)

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc dnn)

add_library(person_postprocessing STATIC
    src/person_postprocessing.cpp
)

target_include_directories(person_postprocessing PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(person_postprocessing PUBLIC
    ${OpenCV_LIBS}
)

target_compile_options(person_postprocessing PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

add_library(person_visualization STATIC
    common/src/person_visualization.cpp
)

target_include_directories(person_visualization PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/common/include"
)

target_link_libraries(person_visualization PUBLIC
    ${OpenCV_LIBS}
)

target_compile_options(person_visualization PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

add_library(person_detector STATIC
    src/person_detector.cpp
)

target_include_directories(person_detector PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(person_detector PUBLIC
    onnxruntime
    person_postprocessing
    ${OpenCV_LIBS}
)

target_compile_options(person_detector PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

add_executable(detect_person_image
    src/detect_person_image.cpp
)

target_link_libraries(detect_person_image PRIVATE
    person_detector
    person_visualization
)

target_compile_options(detect_person_image PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

set_target_properties(detect_person_image PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
)

add_executable(evaluate_coco_person
    src/evaluate_coco_person.cpp
)

target_link_libraries(evaluate_coco_person PRIVATE person_detector)

target_compile_options(evaluate_coco_person PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

set_target_properties(evaluate_coco_person PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
)

add_executable(verify_yolo11n_onnx
    src/verify_yolo11n_onnx.cpp
)

target_link_libraries(verify_yolo11n_onnx PRIVATE onnxruntime)

set_target_properties(verify_yolo11n_onnx PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
)

target_compile_options(verify_yolo11n_onnx PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

include(CTest)

if(BUILD_TESTING)
    add_executable(person_postprocessing_test
        tests/person_postprocessing_test.cpp
    )

    target_link_libraries(person_postprocessing_test PRIVATE
        person_postprocessing
    )

    target_compile_options(person_postprocessing_test PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )

    add_test(
        NAME person_postprocessing_test
        COMMAND person_postprocessing_test
    )
endif()
