add_mlir_library(LMIRToLinalg
    LMIRToLinalg.cpp

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRLinalgDialect
    MLIRArithDialect
    MLIRTensorDialect
    MLIRFuncDialect
    LMIRDialect
)
