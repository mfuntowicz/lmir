#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "lmir/Dialect/LMIR/LMIRDialect.h"
#include "lmir/Conversion/LMIRToLinalg/LMIRToLinalg.h"

int main(int argc, char **argv) {
    mlir::DialectRegistry registry;

    // Register only the upstream dialects we need for our lowering pipeline
    registry.insert<
        mlir::arith::ArithDialect,
        mlir::func::FuncDialect,
        mlir::linalg::LinalgDialect,
        mlir::tensor::TensorDialect,
        mlir::scf::SCFDialect
    >();

    // Register our custom dialect
    registry.insert<lmir::LMIRDialect>();

    // Register conversion passes
    mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
        return hfir::createConvertHFIRToLinalgPass();
    });

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "HFIR optimizer driver\n", registry));
}