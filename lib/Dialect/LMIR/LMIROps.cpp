//
// Created by mfuntowicz on 2/6/26.
//
//===- HFIROps.cpp - HFIR dialect ops implementation ----------------------===//

#include "lmir/Dialect/LMIR/LMIROps.h"

#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace hfir;

#define GET_OP_CLASSES
#include "lmir/Dialect/LMIR/HFIROps.cpp.inc"

//===----------------------------------------------------------------------===//
// FuncOp — custom assembly format (reuses MLIR's function parsing helpers)
//===----------------------------------------------------------------------===//

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false,
      getFunctionTypeAttrName(), getArgAttrsAttrName(), getResAttrsAttrName());
}

//===----------------------------------------------------------------------===//
// FuncOp — builder
//===----------------------------------------------------------------------===//

void FuncOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                   FunctionType type, ArrayRef<NamedAttribute> attrs,
                   ArrayRef<DictionaryAttr> argAttrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name),
                     TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();

  if (argAttrs.empty())
    return;
  assert(type.getNumInputs() == argAttrs.size());
  call_interface_impl::addArgAndResultAttrs(
      builder, state, argAttrs, /*resultAttrs=*/{},
      getArgAttrsAttrName(state.name), getResAttrsAttrName(state.name));
}

//===----------------------------------------------------------------------===//
// CallOp — CallOpInterface
//===----------------------------------------------------------------------===//

auto CallOp::getCallableForCallee() -> CallInterfaceCallable
{
  return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

auto CallOp::setCalleeFromCallable(CallInterfaceCallable callee) -> void
{
  (*this)->setAttr("callee", cast<SymbolRefAttr>(callee));
}

OperandRange CallOp::getArgOperands() { return getOperands(); }

MutableOperandRange CallOp::getArgOperandsMutable() {
  return getOperandsMutable();
}