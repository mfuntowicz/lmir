# TableGen targets for dialect and ops
set(LLVM_TARGET_DEFINITIONS LMIROps.td)

mlir_tablegen(LMIROps.h.inc -gen-op-decls)
mlir_tablegen(LMIROps.cpp.inc -gen-op-defs)
mlir_tablegen(LMIRDialect.h.inc -gen-dialect-decls -dialect=LMIR)
mlir_tablegen(LMIRDialect.cpp.inc -gen-dialect-defs -dialect=LMIR)

add_public_tablegen_target(LMIRIncGen)