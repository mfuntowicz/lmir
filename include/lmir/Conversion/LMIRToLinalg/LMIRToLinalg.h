//===- HFIRToLinalg.h - HFIR to Linalg conversion --------------*- C++ -*-===//
//
// Conversion pass from HFIR dialect to Linalg dialect.
//
//===----------------------------------------------------------------------===//

#ifndef HFIR_CONVERSION_HFIRTOLINALG_H
#define HFIR_CONVERSION_HFIRTOLINALG_H

#include <memory>

namespace mlir {
class Pass;
class RewritePatternSet;
class TypeConverter;
} // namespace mlir

namespace hfir {

/// Populate patterns to convert HFIR ops to Linalg.
void populateHFIRToLinalgConversionPatterns(mlir::RewritePatternSet &patterns);

/// Create a pass to convert HFIR ops to Linalg.
std::unique_ptr<mlir::Pass> createConvertHFIRToLinalgPass();

} // namespace hfir

#endif // HFIR_CONVERSION_HFIRTOLINALG_H
