//===- HFIROps.h - HFIR dialect ops declaration -----------------*- C++ -*-===//

#ifndef HFIR_DIALECT_HFIR_HFIROPS_H
#define HFIR_DIALECT_HFIR_HFIROPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "lmir/Dialect/LMIR/LMIRDialect.h"

using llvm::ArrayRef;
using llvm::StringRef;
using mlir::DictionaryAttr;
using mlir::FunctionType;
using mlir::NamedAttribute;
using mlir::Region;
using mlir::Type;

#define GET_OP_CLASSES
#include "lmir/Dialect/LMIR/HFIROps.h.inc"

#endif // HFIR_DIALECT_HFIR_HFIROPS_H