add_mlir_dialect_library(LMIRDialect
        LMIRDialect.cpp
        LMIROps.cpp

        DEPENDS
        LMIRIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRArithDialect
        MLIRFuncDialect
        MLIRFunctionInterfaces
        MLIRCallInterfaces
        MLIRSideEffectInterfaces
        MLIRInferTypeOpInterface
        MLIRSupport
)