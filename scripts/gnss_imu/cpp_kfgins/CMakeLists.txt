cmake_minimum_required(VERSION 3.14)
project(kfgins_py)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Match the C++ binary's compilation: no optimization flags.
# pybind11 defaults to -O3 which changes floating-point behavior
# (fused multiply-add, reordering) — over 300K+ IMU integration steps
# this produces visibly different trajectories on GNSS-degraded sequences.
set(CMAKE_CXX_FLAGS_RELEASE "-O0")
set(CMAKE_CXX_FLAGS "-O0 -Wno-unused-result")

# KF-GINS source
set(KFGINS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../external/KF-GINS")

# Eigen3 (vendored with KF-GINS)
include_directories(${KFGINS_ROOT}/ThirdParty/eigen-3.3.9)
include_directories(${KFGINS_ROOT}/src)

# yaml-cpp (needed by KF-GINS for config parsing — but we don't use the
# config parser in the Python wrapper, so we can skip it if linking fails)
# Only core filter — skip file I/O (uses abseil which is complex to link).
# Data is fed from Python, not from files.
set(KFGINS_SOURCES
    ${KFGINS_ROOT}/src/kf-gins/gi_engine.cpp
    ${KFGINS_ROOT}/src/kf-gins/insmech.cpp
)

# pybind11
find_package(pybind11 REQUIRED)

pybind11_add_module(kfgins_py
    kfgins_pybind.cpp
    ${KFGINS_SOURCES}
)

# yaml-cpp (needed by fileloader/filesaver — link if available)
find_path(YAMLCPP_INCLUDE yaml-cpp/yaml.h
    PATHS ${KFGINS_ROOT}/ThirdParty/yaml-cpp-0.7.0/include)
if(YAMLCPP_INCLUDE)
    include_directories(${YAMLCPP_INCLUDE})
    # Build yaml-cpp as part of this project
    add_subdirectory(${KFGINS_ROOT}/ThirdParty/yaml-cpp-0.7.0
                     ${CMAKE_BINARY_DIR}/yaml-cpp EXCLUDE_FROM_ALL)
    target_link_libraries(kfgins_py PRIVATE yaml-cpp)
endif()

# abseil (used by KF-GINS logging — try to link if available)
find_path(ABSEIL_INCLUDE absl/strings/str_format.h
    PATHS ${KFGINS_ROOT}/ThirdParty/abseil-cpp-20220623.1)
if(ABSEIL_INCLUDE)
    include_directories(${ABSEIL_INCLUDE})
endif()

# Install location: put the .so next to the Python scripts
set_target_properties(kfgins_py PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
