cmake_minimum_required(VERSION 3.15)
project(dr_ekf_cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Eigen (header-only, fetch if not found) ---
find_package(Eigen3 3.4 QUIET)
if(NOT Eigen3_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(eigen)
endif()

# --- pybind11 ---
find_package(pybind11 REQUIRED)

# --- MOSEK (optional) ---
option(ENABLE_MOSEK "Enable MOSEK SDP solver" OFF)

set(MOSEK_DIR "" CACHE PATH "Path to MOSEK platform directory")
if(NOT MOSEK_DIR)
    if(DEFINED ENV{MOSEK_DIR})
        set(MOSEK_DIR $ENV{MOSEK_DIR})
    endif()
endif()

set(DR_EKF_SOURCES
    src/dynamics.cpp
    src/kalman_utils.cpp
    src/fw_oracle.cpp
    src/ekf.cpp
    src/dr_ekf_cdc.cpp
    src/dr_ekf_tac.cpp
    src/dr_mmse_tac.cpp
    src/dr_mmse_cdc.cpp
    src/bindings.cpp
)

set(DR_EKF_INCLUDE_DIRS include)
set(DR_EKF_LINK_LIBS Eigen3::Eigen)

if(ENABLE_MOSEK)
    find_path(MOSEK_INCLUDE_DIR mosek.h PATHS ${MOSEK_DIR}/h)
    find_library(MOSEK_LIBRARY NAMES mosek64_10_1 mosek64 PATHS ${MOSEK_DIR}/bin ${MOSEK_DIR}/lib)

    if(NOT MOSEK_INCLUDE_DIR OR NOT MOSEK_LIBRARY)
        message(FATAL_ERROR "MOSEK not found. Set MOSEK_DIR to the platform directory, e.g.:\n"
            "  cmake .. -DENABLE_MOSEK=ON -DMOSEK_DIR=/path/to/mosek/10.1/tools/platform/linux64x86")
    endif()

    message(STATUS "MOSEK include: ${MOSEK_INCLUDE_DIR}")
    message(STATUS "MOSEK library: ${MOSEK_LIBRARY}")

    list(APPEND DR_EKF_SOURCES src/mosek_sdp_solver.cpp)
    list(APPEND DR_EKF_INCLUDE_DIRS ${MOSEK_INCLUDE_DIR})
    list(APPEND DR_EKF_LINK_LIBS ${MOSEK_LIBRARY})
else()
    message(STATUS "MOSEK disabled -- building with FW solvers only")
endif()

# --- Build the Python module ---
pybind11_add_module(dr_ekf_cpp ${DR_EKF_SOURCES})

target_include_directories(dr_ekf_cpp PRIVATE ${DR_EKF_INCLUDE_DIRS})
target_link_libraries(dr_ekf_cpp PRIVATE ${DR_EKF_LINK_LIBS})

if(ENABLE_MOSEK)
    target_compile_definitions(dr_ekf_cpp PRIVATE DR_EKF_HAS_MOSEK)
endif()

# Optimization flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(dr_ekf_cpp PRIVATE -O3 -march=native -DNDEBUG)
    if(MINGW)
        target_link_options(dr_ekf_cpp PRIVATE -static-libgcc -static-libstdc++)
    endif()
elseif(MSVC)
    target_compile_options(dr_ekf_cpp PRIVATE /O2 /DNDEBUG)
endif()

# Install the module next to the Python estimator package
install(TARGETS dr_ekf_cpp DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/..)
