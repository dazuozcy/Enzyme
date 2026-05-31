//===- TransformUtils.cpp - Constraint transforms for HMC ------* C++ -*-===//
//
// This file implements constraint transforms for HMC inference.
//
// Reference:
// https://github.com/pyro-ppl/numpyro/blob/master/numpyro/distributions/transforms.py
//
//===----------------------------------------------------------------------===//

#include "TransformUtils.h"

#include "Dialect/Impulse/Impulse.h"
#include "Dialect/Ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include <cmath>

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzyme::transforms;

Value transforms::createLogit(OpBuilder &builder, Location loc, Value x) {
  auto xType = cast<RankedTensorType>(x.getType());
  auto elemType = xType.getElementType();

  auto oneConst = builder.create<arith::ConstantOp>(
      loc, xType,
      DenseElementsAttr::get(xType, builder.getFloatAttr(elemType, 1.0)));
  auto oneMinusX = builder.create<arith::SubFOp>(loc, oneConst, x);
  auto logX = builder.create<math::LogOp>(loc, x);
  auto logOneMinusX = builder.create<math::LogOp>(loc, oneMinusX);
  return builder.create<arith::SubFOp>(loc, logX, logOneMinusX);
}

Value transforms::createLogSigmoid(OpBuilder &builder, Location loc, Value x) {
  auto xType = cast<RankedTensorType>(x.getType());
  auto elemType = xType.getElementType();

  // log_sigmoid(x) = -softplus(-x) = -log_add_exp(-x, 0)
  auto negX = builder.create<arith::NegFOp>(loc, x);
  auto zeroConst = builder.create<arith::ConstantOp>(
      loc, xType,
      DenseElementsAttr::get(xType, builder.getFloatAttr(elemType, 0.0)));
  auto softplusNegX =
      builder.create<impulse::LogAddExpOp>(loc, xType, negX, zeroConst);
  return builder.create<arith::NegFOp>(loc, softplusNegX);
}

int64_t transforms::getUnconstrainedSize(int64_t constrainedSize,
                                         impulse::SupportKind kind) {
  switch (kind) {
  case impulse::SupportKind::REAL:
  case impulse::SupportKind::POSITIVE:
  case impulse::SupportKind::UNIT_INTERVAL:
  case impulse::SupportKind::INTERVAL:
  case impulse::SupportKind::GREATER_THAN:
  case impulse::SupportKind::LESS_THAN:
    return constrainedSize;
  }
  llvm_unreachable("Unknown SupportKind");
}

int64_t transforms::getConstrainedSize(int64_t unconstrainedSize,
                                       impulse::SupportKind kind) {
  switch (kind) {
  case impulse::SupportKind::REAL:
  case impulse::SupportKind::POSITIVE:
  case impulse::SupportKind::UNIT_INTERVAL:
  case impulse::SupportKind::INTERVAL:
  case impulse::SupportKind::GREATER_THAN:
  case impulse::SupportKind::LESS_THAN:
    return unconstrainedSize;
  }
  llvm_unreachable("Unknown SupportKind");
}

Value transforms::unconstrain(OpBuilder &builder, Location loc,
                              Value constrained, impulse::SupportAttr support) {
  auto kind = support.getKind();
  auto xType = cast<RankedTensorType>(constrained.getType());
  auto elemType = xType.getElementType();

  switch (kind) {
  case impulse::SupportKind::REAL:
    // Identity
    return constrained;
  case impulse::SupportKind::POSITIVE:
    return builder.create<math::LogOp>(loc, constrained);
  case impulse::SupportKind::UNIT_INTERVAL:
    return createLogit(builder, loc, constrained);
  case impulse::SupportKind::INTERVAL: {
    // z = logit((x - a) / (b - a))
    auto lowerAttr = support.getLowerBound();
    auto upperAttr = support.getUpperBound();
    if (!lowerAttr || !upperAttr) {
      llvm_unreachable("INTERVAL support requires lower and upper bounds");
    }
    double lower = lowerAttr.getValueAsDouble();
    double upper = upperAttr.getValueAsDouble();

    auto lowerConst = builder.create<arith::ConstantOp>(
        loc, xType,
        DenseElementsAttr::get(xType, builder.getFloatAttr(elemType, lower)));
    auto scaleConst = builder.create<arith::ConstantOp>(
        loc, xType,
        DenseElementsAttr::get(xType,
                               builder.getFloatAttr(elemType, upper - lower)));

    auto shifted = builder.create<arith::SubFOp>(loc, constrained, lowerConst);
    auto normalized = builder.create<arith::DivFOp>(loc, shifted, scaleConst);
    return createLogit(builder, loc, normalized);
  }
  case impulse::SupportKind::GREATER_THAN: {
    // z = log(x - lower)
    auto lowerAttr = support.getLowerBound();
    if (!lowerAttr) {
      llvm_unreachable("GREATER_THAN support requires lower bound");
    }
    double lower = lowerAttr.getValueAsDouble();

    auto lowerConst = builder.create<arith::ConstantOp>(
        loc, xType,
        DenseElementsAttr::get(xType, builder.getFloatAttr(elemType, lower)));
    auto shifted = builder.create<arith::SubFOp>(loc, constrained, lowerConst);
    return builder.create<math::LogOp>(loc, shifted);
  }
  case impulse::SupportKind::LESS_THAN: {
    // z = log(upper - x)
    auto upperAttr = support.getUpperBound();
    if (!upperAttr) {
      llvm_unreachable("LESS_THAN support requires upper bound");
    }
    double upper = upperAttr.getValueAsDouble();

    auto upperConst = builder.create<arith::ConstantOp>(
        loc, xType,
        DenseElementsAttr::get(xType, builder.getFloatAttr(elemType, upper)));
    auto shifted = builder.create<arith::SubFOp>(loc, upperConst, constrained);
    return builder.create<math::LogOp>(loc, shifted);
  }
  }
  llvm_unreachable("Unknown SupportKind");
}

Value transforms::constrain(OpBuilder &builder, Location loc,
                            Value unconstrained, impulse::SupportAttr support) {
  auto kind = support.getKind();
  auto zType = cast<RankedTensorType>(unconstrained.getType());
  auto elemType = zType.getElementType();

  switch (kind) {
  case impulse::SupportKind::REAL:
    // Identity
    return unconstrained;
  case impulse::SupportKind::POSITIVE:
    return builder.create<math::ExpOp>(loc, unconstrained);
  case impulse::SupportKind::UNIT_INTERVAL:
    // x = sigmoid(z)
    return builder.create<impulse::LogisticOp>(loc, unconstrained.getType(),
                                       unconstrained);
  case impulse::SupportKind::INTERVAL: {
    // x = a + (b - a) * sigmoid(z)
    auto lowerAttr = support.getLowerBound();
    auto upperAttr = support.getUpperBound();
    if (!lowerAttr || !upperAttr) {
      llvm_unreachable("INTERVAL support requires lower and upper bounds");
    }
    double lower = lowerAttr.getValueAsDouble();
    double upper = upperAttr.getValueAsDouble();

    auto lowerConst = builder.create<arith::ConstantOp>(
        loc, zType,
        DenseElementsAttr::get(zType, builder.getFloatAttr(elemType, lower)));
    auto scaleConst = builder.create<arith::ConstantOp>(
        loc, zType,
        DenseElementsAttr::get(zType,
                               builder.getFloatAttr(elemType, upper - lower)));

    auto sigmoid = builder.create<impulse::LogisticOp>(
        loc, unconstrained.getType(), unconstrained);
    auto scaled = builder.create<arith::MulFOp>(loc, scaleConst, sigmoid);
    return builder.create<arith::AddFOp>(loc, lowerConst, scaled);
  }
  case impulse::SupportKind::GREATER_THAN: {
    // x = lower + exp(z)
    auto lowerAttr = support.getLowerBound();
    if (!lowerAttr) {
      llvm_unreachable("GREATER_THAN support requires lower bound");
    }
    double lower = lowerAttr.getValueAsDouble();

    auto lowerConst = builder.create<arith::ConstantOp>(
        loc, zType,
        DenseElementsAttr::get(zType, builder.getFloatAttr(elemType, lower)));
    auto expZ = builder.create<math::ExpOp>(loc, unconstrained);
    return builder.create<arith::AddFOp>(loc, lowerConst, expZ);
  }
  case impulse::SupportKind::LESS_THAN: {
    // x = upper - exp(z)
    auto upperAttr = support.getUpperBound();
    if (!upperAttr) {
      llvm_unreachable("LESS_THAN support requires upper bound");
    }
    double upper = upperAttr.getValueAsDouble();

    auto upperConst = builder.create<arith::ConstantOp>(
        loc, zType,
        DenseElementsAttr::get(zType, builder.getFloatAttr(elemType, upper)));
    auto expZ = builder.create<math::ExpOp>(loc, unconstrained);
    return builder.create<arith::SubFOp>(loc, upperConst, expZ);
  }
  }

  llvm_unreachable("Unknown SupportKind");
}

Value transforms::logAbsDetJacobian(OpBuilder &builder, Location loc,
                                    Value unconstrained,
                                    impulse::SupportAttr support) {
  auto kind = support.getKind();
  auto zType = cast<RankedTensorType>(unconstrained.getType());
  auto elemType = zType.getElementType();
  auto scalarType = RankedTensorType::get({}, elemType);

  switch (kind) {
  case impulse::SupportKind::REAL: {
    // Identity: log|det(I)| = 0
    return builder.create<arith::ConstantOp>(
        loc, scalarType,
        DenseElementsAttr::get(scalarType,
                               builder.getFloatAttr(elemType, 0.0)));
  }
  case impulse::SupportKind::POSITIVE: {
    // x = exp(z), dx/dz = exp(z)
    // log|det(J)| = sum(log|dx_i/dz_i|) = sum(z)
    auto ones = builder.create<arith::ConstantOp>(
        loc, zType,
        DenseElementsAttr::get(zType, builder.getFloatAttr(elemType, 1.0)));
    return builder.create<impulse::DotOp>(
        loc, scalarType, unconstrained, ones,
        builder.getDenseI64ArrayAttr({}), builder.getDenseI64ArrayAttr({}),
        builder.getDenseI64ArrayAttr({0}), builder.getDenseI64ArrayAttr({0}));
  }
  case impulse::SupportKind::UNIT_INTERVAL: {
    // x = sigmoid(z), dx/dz = sigmoid(z) * (1 - sigmoid(z))
    // log|det(J)| = sum(log(sigmoid(z)) + log(1 - sigmoid(z)))
    //             = sum(log_sigmoid(z) + log_sigmoid(-z))
    auto logSigZ = createLogSigmoid(builder, loc, unconstrained);
    auto negZ = builder.create<arith::NegFOp>(loc, unconstrained);
    auto logSigNegZ = createLogSigmoid(builder, loc, negZ);
    auto logProduct = builder.create<arith::AddFOp>(loc, logSigZ, logSigNegZ);
    auto ones = builder.create<arith::ConstantOp>(
        loc, zType,
        DenseElementsAttr::get(zType, builder.getFloatAttr(elemType, 1.0)));
    return builder.create<impulse::DotOp>(
        loc, scalarType, logProduct, ones,
        builder.getDenseI64ArrayAttr({}), builder.getDenseI64ArrayAttr({}),
        builder.getDenseI64ArrayAttr({0}), builder.getDenseI64ArrayAttr({0}));
  }
  case impulse::SupportKind::INTERVAL: {
    // log|det(J)| = sum(log_sigmoid(z) + log(1 - sigmoid(z))) + n*log(scale)
    auto lowerAttr = support.getLowerBound();
    auto upperAttr = support.getUpperBound();
    if (!lowerAttr || !upperAttr) {
      llvm_unreachable("INTERVAL support requires lower and upper bounds");
    }
    double scale = upperAttr.getValueAsDouble() - lowerAttr.getValueAsDouble();

    auto logSigZ = createLogSigmoid(builder, loc, unconstrained);
    auto negZ = builder.create<arith::NegFOp>(loc, unconstrained);
    auto logSigNegZ = createLogSigmoid(builder, loc, negZ);
    auto logProduct = builder.create<arith::AddFOp>(loc, logSigZ, logSigNegZ);
    auto ones = builder.create<arith::ConstantOp>(
        loc, zType,
        DenseElementsAttr::get(zType, builder.getFloatAttr(elemType, 1.0)));
    auto sumLogProduct = builder.create<impulse::DotOp>(
        loc, scalarType, logProduct, ones,
        builder.getDenseI64ArrayAttr({}), builder.getDenseI64ArrayAttr({}),
        builder.getDenseI64ArrayAttr({0}), builder.getDenseI64ArrayAttr({0}));

    int64_t n = zType.getNumElements();
    double logScaleTerm = n * std::log(scale);
    auto logScaleConst = builder.create<arith::ConstantOp>(
        loc, scalarType,
        DenseElementsAttr::get(scalarType,
                               builder.getFloatAttr(elemType, logScaleTerm)));
    return builder.create<arith::AddFOp>(loc, sumLogProduct, logScaleConst);
  }
  case impulse::SupportKind::GREATER_THAN:
  case impulse::SupportKind::LESS_THAN: {
    // log|det(J)| = sum(z)
    auto ones = builder.create<arith::ConstantOp>(
        loc, zType,
        DenseElementsAttr::get(zType, builder.getFloatAttr(elemType, 1.0)));
    return builder.create<impulse::DotOp>(
        loc, scalarType, unconstrained, ones,
        builder.getDenseI64ArrayAttr({}), builder.getDenseI64ArrayAttr({}),
        builder.getDenseI64ArrayAttr({0}), builder.getDenseI64ArrayAttr({0}));
  }
  }

  llvm_unreachable("Unknown SupportKind");
}
