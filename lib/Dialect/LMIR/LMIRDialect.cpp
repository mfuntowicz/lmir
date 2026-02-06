//===- HFIRDialect.cpp - HFIR dialect implementation ----------------------===//

#include "hfir/Dialect/LMIR/LMIRDialect.h"
#include "hfir/Dialect/LMIR/LMIROps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
using namespace hfir;

#include "hfir/Dialect/LMIR/HFIRDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// HFIR Dialect
//===----------------------------------------------------------------------===//

void HFIRDialect::initialize() {
    addOperations<
  #define GET_OP_LIST
  #include "hfir/Dialect/LMIR/HFIROps.cpp.inc"

    >();
}