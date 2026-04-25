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
)
target_include_directories(lab PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(lab PRIVATE ${OpenCV_LIBS})

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
