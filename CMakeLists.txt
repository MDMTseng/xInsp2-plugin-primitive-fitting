cmake_minimum_required(VERSION 3.16)
project(primitive_fitting CXX C)
set(CMAKE_CXX_STANDARD 20)

# XINSP2_ROOT — env var preferred; auto-detect a sibling xInsp2 checkout
# as a fallback so this builds without ceremony if both repos sit under a
# common parent (the typical layout: xInsp/{xInsp2,plugins}/ ).
if(NOT DEFINED XINSP2_ROOT AND DEFINED ENV{XINSP2_ROOT})
    set(XINSP2_ROOT $ENV{XINSP2_ROOT})
endif()
if(NOT DEFINED XINSP2_ROOT)
    foreach(rel "../../xInsp2" "../../../xInsp2" "../../../../xInsp2")
        get_filename_component(_p "${CMAKE_CURRENT_SOURCE_DIR}/${rel}" ABSOLUTE)
        if(EXISTS "${_p}/sdk/cmake/xinsp2_plugin.cmake")
            set(XINSP2_ROOT ${_p})
            break()
        endif()
    endforeach()
endif()
include(${XINSP2_ROOT}/sdk/cmake/xinsp2_plugin.cmake)

find_package(OpenCV REQUIRED)

xinsp2_add_plugin(primitive_fitting primitive_fitting.cpp)
target_link_libraries(primitive_fitting PRIVATE ${OpenCV_LIBS})
target_include_directories(primitive_fitting PRIVATE ${OpenCV_INCLUDE_DIRS})
if(MSVC)
    target_compile_options(primitive_fitting PRIVATE /utf-8 /wd4244 /wd4267)
    target_compile_definitions(primitive_fitting PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
endif()

# --- Native test (loads the DLL via LoadLibrary, runs baseline + custom)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_native.cpp)
    xinsp2_add_plugin_test(primitive_fitting_test tests/test_native.cpp)
    target_compile_definitions(primitive_fitting_test PRIVATE
        PRIMITIVE_FITTING_DLL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/primitive_fitting.dll")
    add_dependencies(primitive_fitting_test primitive_fitting)
    enable_testing()
    add_test(NAME primitive_fitting_test
             COMMAND primitive_fitting_test
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()
