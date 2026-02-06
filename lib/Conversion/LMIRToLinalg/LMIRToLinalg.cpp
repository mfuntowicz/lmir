//===- HFIRToLinalg.cpp - HFIR to Linalg conversion ----------------------===//
//
// This file implements lowering of HFIR ops to Linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "lmir/Conversion/LMIRToLinalg/LMIRToLinalg.h"
#include "lmir/Dialect/LMIR/LMIROps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace hfir;

namespace {

//===----------------------------------------------------------------------===//
// Binary elementwise ops lowering (add, sub, mul, div)
//===----------------------------------------------------------------------===//

template <typename HFIROp, typename ArithOp>
struct BinaryElementwiseOpConversion : public OpConversionPattern<HFIROp> {
  using OpConversionPattern<HFIROp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(HFIROp op, typename HFIROp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());

    if (!resultType) {
      // Scalar case - lower directly to arith op
      rewriter.replaceOpWithNewOp<ArithOp>(op, adaptor.getLhs(), adaptor.getRhs());
      return success();
    }

    // Tensor case - lower to linalg.map
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    // Create empty tensor for output
    SmallVector<Value> dynDims;
    for (auto [idx, dim] : llvm::enumerate(resultType.getShape())) {
      if (ShapedType::isDynamic(dim)) {
        Value dimValue = rewriter.create<tensor::DimOp>(loc, lhs, idx);
        dynDims.push_back(dimValue);
      }
    }
    Value init = rewriter.create<tensor::EmptyOp>(
        loc, resultType, dynDims);

    // Create linalg.map with binary operation
    auto mapOp = rewriter.create<linalg::MapOp>(
        loc, ValueRange{lhs, rhs}, init,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value result = b.create<ArithOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, result);
        });

    rewriter.replaceOp(op, mapOp.getResult()[0]);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Unary elementwise ops lowering (relu, neg)
//===----------------------------------------------------------------------===//

struct ReluOpConversion : public OpConversionPattern<ReluOp> {
  using OpConversionPattern<ReluOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());

    if (!resultType) {
      // Scalar case - max(input, 0)
      Value zero = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getZeroAttr(adaptor.getInput().getType()));
      rewriter.replaceOpWithNewOp<arith::MaximumFOp>(op, adaptor.getInput(), zero);
      return success();
    }

    // Tensor case - lower to linalg.map
    Value input = adaptor.getInput();

    SmallVector<Value> dynDims;
    for (auto [idx, dim] : llvm::enumerate(resultType.getShape())) {
      if (ShapedType::isDynamic(dim)) {
        Value dimValue = rewriter.create<tensor::DimOp>(loc, input, idx);
        dynDims.push_back(dimValue);
      }
    }
    Value init = rewriter.create<tensor::EmptyOp>(loc, resultType, dynDims);

    auto mapOp = rewriter.create<linalg::MapOp>(
        loc, input, init,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value zero = b.create<arith::ConstantOp>(
              loc, b.getZeroAttr(resultType.getElementType()));
          Value result = b.create<arith::MaximumFOp>(loc, args[0], zero);
          b.create<linalg::YieldOp>(loc, result);
        });

    rewriter.replaceOp(op, mapOp.getResult()[0]);
    return success();
  }
};

struct NegOpConversion : public OpConversionPattern<NegOp> {
  using OpConversionPattern<NegOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(NegOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());

    if (!resultType) {
      // Scalar case
      rewriter.replaceOpWithNewOp<arith::NegFOp>(op, adaptor.getInput());
      return success();
    }

    // Tensor case - lower to linalg.map
    Value input = adaptor.getInput();

    SmallVector<Value> dynDims;
    for (auto [idx, dim] : llvm::enumerate(resultType.getShape())) {
      if (ShapedType::isDynamic(dim)) {
        Value dimValue = rewriter.create<tensor::DimOp>(loc, input, idx);
        dynDims.push_back(dimValue);
      }
    }
    Value init = rewriter.create<tensor::EmptyOp>(loc, resultType, dynDims);

    auto mapOp = rewriter.create<linalg::MapOp>(
        loc, input, init,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value result = b.create<arith::NegFOp>(loc, args[0]);
          b.create<linalg::YieldOp>(loc, result);
        });

    rewriter.replaceOp(op, mapOp.getResult()[0]);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// MatMul lowering
//===----------------------------------------------------------------------===//

struct MatMulOpConversion : public OpConversionPattern<MatMulOp> {
  using OpConversionPattern<MatMulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MatMulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    // Create empty tensor for output
    SmallVector<Value> dynDims;
    for (auto [idx, dim] : llvm::enumerate(resultType.getShape())) {
      if (ShapedType::isDynamic(dim)) {
        Value dimValue = rewriter.create<tensor::DimOp>(loc, lhs, idx);
        dynDims.push_back(dimValue);
      }
    }
    Value init = rewriter.create<tensor::EmptyOp>(loc, resultType, dynDims);

    // Fill with zeros
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(resultType.getElementType()));
    Value filled = rewriter.create<linalg::FillOp>(loc, zero, init).getResult(0);

    // Create linalg.matmul
    rewriter.replaceOpWithNewOp<linalg::MatmulOp>(
        op, ValueRange{lhs, rhs}, ValueRange{filled});

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Conversion pass
//===----------------------------------------------------------------------===//

class ConvertHFIRToLinalgPass
    : public PassWrapper<ConvertHFIRToLinalgPass, OperationPass<>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertHFIRToLinalgPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, arith::ArithDialect,
                    tensor::TensorDialect, func::FuncDialect>();
  }

  StringRef getArgument() const final { return "convert-hfir-to-linalg"; }
  StringRef getDescription() const final {
    return "Convert HFIR ops to Linalg ops";
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = &getContext();

    ConversionTarget target(*context);
    target.addLegalDialect<linalg::LinalgDialect, arith::ArithDialect,
                           tensor::TensorDialect, func::FuncDialect>();
    target.addIllegalDialect<HFIRDialect>();
    target.addLegalOp<FuncOp, ReturnOp>();

    RewritePatternSet patterns(context);
    populateHFIRToLinalgConversionPatterns(patterns);

    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void hfir::populateHFIRToLinalgConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<
      BinaryElementwiseOpConversion<AddOp, arith::AddFOp>,
      BinaryElementwiseOpConversion<SubOp, arith::SubFOp>,
      BinaryElementwiseOpConversion<MulOp, arith::MulFOp>,
      BinaryElementwiseOpConversion<DivOp, arith::DivFOp>,
      ReluOpConversion,
      NegOpConversion,
      MatMulOpConversion
  >(patterns.getContext());
}

std::unique_ptr<mlir::Pass> hfir::createConvertHFIRToLinalgPass() {
  return std::make_unique<ConvertHFIRToLinalgPass>();
}
