cmake_minimum_required(VERSION 3.16)
project(primitive_fitting_lab CXX)
set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)

if(MSVC)
    add_compile_options(/utf-8 /O2 /W3 /wd4244 /wd4267 /openmp)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX)
endif()

add_executable(lab
    main.cpp
    scene.cpp
    eval.cpp
    algo_naive.cpp
    algo_dp.cpp
    algo_dijkstra.cpp
    algo_tensor_voting.cpp
    algo_caliper_ransac.cpp
    algo_caliper_ransac_variants.cpp
    algo_caliper_dp.cpp
    algo_caliper_dp_poly.cpp
    algo_subregion_tv_peaks.cpp
    algo_subregion_tv_band.cpp
    algo_subregion_tv_strips.cpp
    algo_subregion_dp_strips.cpp
    algo_spline_knot_dp.cpp
    algo_constrained_grid.cpp
    algo_constrained_ransac.cpp
    algo_constrained_knot_dp.cpp
    algo_caliper_cnn.cpp
    algo_caliper_cnn_ort.cpp
)
target_include_directories(lab PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(lab PRIVATE ${OpenCV_LIBS})

# Optional ONNX Runtime backend for the cross-caliper CNN. The
# variable ORT_ROOT may be set on the cmake command line or as an
# env var; the default below points at a NuGet-cached ORT 1.23.2.
# When found, defines XINSP_ENABLE_ORT and links onnxruntime.lib;
# the resulting build also copies onnxruntime.dll next to lab.exe.
if(NOT DEFINED ORT_ROOT AND DEFINED ENV{ORT_ROOT})
    set(ORT_ROOT $ENV{ORT_ROOT})
endif()
if(NOT DEFINED ORT_ROOT)
    set(ORT_ROOT "$ENV{USERPROFILE}/.nuget/packages/microsoft.ml.onnxruntime/1.23.2")
endif()
set(_ORT_INC "${ORT_ROOT}/build/native/include")
set(_ORT_LIB "${ORT_ROOT}/runtimes/win-x64/native/onnxruntime.lib")
set(_ORT_DLL "${ORT_ROOT}/runtimes/win-x64/native/onnxruntime.dll")
if(EXISTS "${_ORT_INC}/onnxruntime_cxx_api.h" AND EXISTS "${_ORT_LIB}")
    target_compile_definitions(lab PRIVATE XINSP_ENABLE_ORT)
    target_include_directories(lab PRIVATE "${_ORT_INC}")
    target_link_libraries(lab PRIVATE "${_ORT_LIB}")
    add_custom_command(TARGET lab POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_ORT_DLL}"
                "$<TARGET_FILE_DIR:lab>/onnxruntime.dll"
        COMMENT "Copying onnxruntime.dll next to lab.exe")
    message(STATUS "Lab: ONNX Runtime enabled (${ORT_ROOT})")
else()
    message(STATUS "Lab: ONNX Runtime NOT found at ${ORT_ROOT} -- algo_caliper_cnn_ort.cpp will stub out")
endif()

# Caliper-ROI dataset dump tool (separate executable).
add_executable(dump_caliper_dataset dump_caliper_dataset.cpp scene.cpp)
target_include_directories(dump_caliper_dataset PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(dump_caliper_dataset PRIVATE ${OpenCV_LIBS})

# /fp:fast is scoped to the tensor-voting translation unit only. Other
# algorithms (e.g. algo_dijkstra) rely on IEEE +inf semantics which
# /fp:fast breaks.
if(MSVC)
    set_source_files_properties(algo_tensor_voting.cpp PROPERTIES
        COMPILE_OPTIONS "/fp:fast")
endif()
