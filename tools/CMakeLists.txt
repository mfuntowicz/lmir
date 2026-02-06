add_llvm_executable(lmir-opt
    lmir-opt.cpp
)

target_link_libraries(lmir-opt
    PRIVATE
    LMIRDialect
    LMIRToLinalg
    MLIRArithDialect
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRTensorDialect
    MLIRSCFDialect
    MLIRPass
    MLIRParser
    MLIROptLib
    MLIRSupport
    MLIRIR
)

target_include_directories(lmir-opt BEFORE PRIVATE
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include
)