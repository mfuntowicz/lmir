# CMakeLists.txt
cmake_minimum_required(VERSION 3.25)
project(lmir
    VERSION 0.1.0
    LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Option: use a pre-installed MLIR or fetch from source
# ---------------------------------------------------------------------------
option(LMIR_USE_SYSTEM_MLIR "Use a pre-installed MLIR (find_package) instead of FetchContent" OFF)

if(LMIR_USE_SYSTEM_MLIR)
    find_package(MLIR REQUIRED CONFIG)
    find_package(LLVM REQUIRED CONFIG)
    message(STATUS "Using pre-installed MLIR: ${MLIR_DIR}")
    message(STATUS "Using pre-installed LLVM: ${LLVM_DIR}")

    set(MLIR_CMAKE_DIR "${MLIR_DIR}")
    set(LLVM_CMAKE_DIR "${LLVM_DIR}")
else()
    include(cmake/FetchLLVM.cmake)
endif()

# ---------------------------------------------------------------------------
# MLIR/LLVM helpers — needed for TableGen, lit, etc.
# ---------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})

# Our own include trees (source + generated)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

# ---------------------------------------------------------------------------
# Subdirectories
# ---------------------------------------------------------------------------
add_subdirectory(include/lmir/Dialect/LMIR)
add_subdirectory(lib)
add_subdirectory(tools)

# Tests (optional, requires lit + FileCheck)
# option(LMIR_ENABLE_TESTS "Build and register lit tests" OFF)
# if(LMIR_ENABLE_TESTS)
#     add_subdirectory(test)
# endif()