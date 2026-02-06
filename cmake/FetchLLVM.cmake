# cmake/FetchLLVM.cmake
#
# Fetches LLVM + MLIR from source and builds only what we need.
# ---------------------------------------------------------------------------
include(FetchContent)

# Pin to LLVM 19.1.0 release
set(HFIR_LLVM_VERSION "llvmorg-21.1.0" CACHE STRING "LLVM Git tag or commit to fetch")

message(STATUS "Fetching LLVM/MLIR @ ${HFIR_LLVM_VERSION} — first build will take a while...")

FetchContent_Declare(
        llvm-project
        GIT_REPOSITORY https://github.com/llvm/llvm-project.git
        GIT_TAG        ${HFIR_LLVM_VERSION}
        GIT_SHALLOW    FALSE  # Need full clone for llvmorg-21.1.0 tag
        GIT_PROGRESS   TRUE
)

# ---------------------------------------------------------------------------
# Minimal LLVM config — only build what MLIR needs
# ---------------------------------------------------------------------------
set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "" FORCE)
set(LLVM_TARGETS_TO_BUILD "host" CACHE STRING "" FORCE)
set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "" FORCE)
set(LLVM_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLVM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLVM_BUILD_TOOLS ON CACHE BOOL "" FORCE)
set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MLIR_ENABLE_BINDINGS_PYTHON OFF CACHE BOOL "" FORCE)

if(NOT llvm-project_POPULATED)
    fetchcontent_makeavailable(llvm-project)
    add_subdirectory(${llvm-project_SOURCE_DIR}/llvm ${llvm-project_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# ---------------------------------------------------------------------------
# Set the variables that the rest of the build expects
# ---------------------------------------------------------------------------
set(MLIR_CMAKE_DIR "${llvm-project_SOURCE_DIR}/mlir/cmake/modules" CACHE PATH "" FORCE)
set(LLVM_CMAKE_DIR "${llvm-project_SOURCE_DIR}/llvm/cmake/modules" CACHE PATH "" FORCE)

set(MLIR_INCLUDE_DIRS
        "${llvm-project_SOURCE_DIR}/mlir/include"
        "${llvm-project_BINARY_DIR}/tools/mlir/include"
        CACHE PATH "" FORCE
)
set(LLVM_INCLUDE_DIRS
        "${llvm-project_SOURCE_DIR}/llvm/include"
        "${llvm-project_BINARY_DIR}/include"
        CACHE PATH "" FORCE
)

set(MLIR_TABLEGEN_EXE mlir-tblgen)