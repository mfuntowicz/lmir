//===- HFIROps.td - HFIR dialect ops -----------------------*- tablegen -*-===//

#ifndef HFIR_OPS_TD
#define HFIR_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// HFIR Dialect
//===----------------------------------------------------------------------===//

def HFIR_Dialect : Dialect {
  let name = "hfir";
  let summary = "High-level Fused IR for ML model compilation";
  let description = [{
    HFIR (High-Fidelity IR) is a dialect designed for expressing ML models
    at a high level of abstraction with a Pythonic functional call syntax.

    All ops use functional notation: `hfir.add(%a, %b)` rather than
    `hfir.add %a, %b`, mirroring PyTorch's `torch.add(a, b)` API.
  }];
  let cppNamespace = "::hfir";
  let dependentDialects = [
    "::mlir::arith::ArithDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Base classes
//===----------------------------------------------------------------------===//

class HFIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<HFIR_Dialect, mnemonic, traits>;

// Type constraint for tensor or scalar (integer or float)
def HFIR_TensorOrScalar : TypeConstraint<
    Or<[AnyRankedTensor.predicate,
        AnySignlessInteger.predicate,
        AnyFloat.predicate]>,
    "ranked tensor or scalar (signless integer or float)">;

// Binary elementwise: hfir.<op>(%lhs, %rhs) : type
class HFIR_BinaryOp<string mnemonic, list<Trait> traits = []> :
    HFIR_Op<mnemonic, !listconcat([Pure, SameOperandsAndResultType], traits)> {
  let arguments = (ins HFIR_TensorOrScalar:$lhs, HFIR_TensorOrScalar:$rhs);
  let results = (outs HFIR_TensorOrScalar:$result);

  // Functional call syntax: hfir.add(%a, %b) : tensor<?x768xf32> or i32
  let assemblyFormat = "`(` $lhs `,` $rhs `)` attr-dict `:` type($result)";
}

// Unary elementwise: hfir.<op>(%input) : type
class HFIR_UnaryOp<string mnemonic, list<Trait> traits = []> :
    HFIR_Op<mnemonic, !listconcat([Pure, SameOperandsAndResultType], traits)> {
  let arguments = (ins HFIR_TensorOrScalar:$input);
  let results = (outs HFIR_TensorOrScalar:$result);

  let assemblyFormat = "`(` $input `)` attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Arithmetic ops — hfir.add(), hfir.sub(), hfir.mul(), hfir.div()
//===----------------------------------------------------------------------===//

def HFIR_AddOp : HFIR_BinaryOp<"add"> {
  let summary = "Elementwise tensor addition";
  let description = [{
    Example:
    ```mlir
    %c = hfir.add(%a, %b) : tensor<?x768xf32>
    ```
  }];
}

def HFIR_SubOp : HFIR_BinaryOp<"sub"> {
  let summary = "Elementwise tensor subtraction";
}

def HFIR_MulOp : HFIR_BinaryOp<"mul"> {
  let summary = "Elementwise tensor multiplication (Hadamard product)";
}

def HFIR_DivOp : HFIR_BinaryOp<"div"> {
  let summary = "Elementwise tensor division";
}

//===----------------------------------------------------------------------===//
// Unary ops — hfir.relu(), hfir.neg()
//===----------------------------------------------------------------------===//

def HFIR_ReluOp : HFIR_UnaryOp<"relu"> {
  let summary = "Rectified linear unit activation";
}

def HFIR_NegOp : HFIR_UnaryOp<"neg"> {
  let summary = "Elementwise negation";
}

//===----------------------------------------------------------------------===//
// MatMul — hfir.matmul(%a, %b) : (T1, T2) -> T3
//===----------------------------------------------------------------------===//

def HFIR_MatMulOp : HFIR_Op<"matmul", [Pure]> {
  let summary = "Matrix multiplication";
  let description = [{
    Example:
    ```mlir
    %out = hfir.matmul(%x, %w) : (tensor<?x768xf32>, tensor<768x768xf32>) -> tensor<?x768xf32>
    ```
  }];

  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    `(` $lhs `,` $rhs `)` attr-dict
    `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// User-defined functions — hfir.func / hfir.return / hfir.call
//===----------------------------------------------------------------------===//

def HFIR_FuncOp : HFIR_Op<"func", [
    AffineScope, AutomaticAllocationScope,
    FunctionOpInterface, IsolatedFromAbove, Symbol
  ]> {
  let summary = "User-defined function (like a PyTorch nn.Module.forward)";
  let description = [{
    Defines a named, callable function. Users express model layers,
    attention heads, custom activations, etc. as `hfir.func` ops.

    Example:
    ```mlir
    hfir.func @my_gelu(%x: tensor<?x768xf32>) -> tensor<?x768xf32> {
      %0 = hfir.mul(%x, %x) : tensor<?x768xf32>
      hfir.return %0 : tensor<?x768xf32>
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );

  let regions = (region AnyRegion:$body);

  let builders = [
    OpBuilder<(ins "StringRef":$name, "FunctionType":$type,
               CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
               CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)>
  ];

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Returns the callable region.
    Region *getCallableRegion() { return &getBody(); }
  }];

  // Use the standard function assembly format (same as func.func)
  let hasCustomAssemblyFormat = 1;
}

def HFIR_ReturnOp : HFIR_Op<"return", [
    Pure, HasParent<"FuncOp">, ReturnLike, Terminator
  ]> {
  let summary = "Return values from an hfir.func";
  let description = [{
    Example:
    ```mlir
    hfir.return %0 : tensor<?x768xf32>
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";

  let builders = [
    OpBuilder<(ins), [{ /* empty return */ }]>
  ];
}

def HFIR_CallOp : HFIR_Op<"call", [
    DeclareOpInterfaceMethods<CallOpInterface>
  ]> {
  let summary = "Call a user-defined hfir.func";
  let description = [{
    Calls a named `hfir.func`. Mirrors PyTorch's calling convention.

    Example:
    ```mlir
    %out = hfir.call @my_gelu(%x) : (tensor<?x768xf32>) -> tensor<?x768xf32>
    ```
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$operands,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, $results)
  }];
}

#endif // HFIR_OPS_TD