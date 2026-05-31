//===- HMCUtils.cpp - Utilities for HMC/NUTS inference -------* C++ -*-===//
//
// This file implements utility functions for Hamiltonian Monte Carlo (HMC) and
// No-U-Turn Sampler (NUTS) implementations.
//
// Reference:
// https://github.com/pyro-ppl/numpyro/blob/master/numpyro/infer/hmc_util.py
//
//===----------------------------------------------------------------------===//

#include "HMCUtils.h"

#include "Dialect/Impulse/Impulse.h"
#include "Dialect/Ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include <cmath>
#include <limits>

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::impulse;

SmallVector<Value> NUTSTreeState::toValues() const {
  return {q_left,        p_left,        grad_left,
          q_right,       p_right,       grad_right,
          q_proposal,    grad_proposal, U_proposal,
          H_proposal,    depth,         weight,
          turning,       diverging,     sum_accept_probs,
          num_proposals, p_sum,         rng};
}

NUTSTreeState NUTSTreeState::fromValues(ArrayRef<Value> values) {
  assert(values.size() == 18 && "Expected 18 NUTSTreeState fields");
  return {.q_left = values[0],
          .p_left = values[1],
          .grad_left = values[2],
          .q_right = values[3],
          .p_right = values[4],
          .grad_right = values[5],
          .q_proposal = values[6],
          .grad_proposal = values[7],
          .U_proposal = values[8],
          .H_proposal = values[9],
          .depth = values[10],
          .weight = values[11],
          .turning = values[12],
          .diverging = values[13],
          .sum_accept_probs = values[14],
          .num_proposals = values[15],
          .p_sum = values[16],
          .rng = values[17]};
}

SmallVector<Type> NUTSTreeState::getTypes() const {
  SmallVector<Type> types;
  for (auto val : toValues())
    types.push_back(val.getType());
  return types;
}

Value impulse::conditionalDump(OpBuilder &builder, Location loc, Value value,
                               StringRef label, bool debugDump) {
  if (debugDump) {
    return builder.create<enzyme::DumpOp>(loc, value.getType(), value,
                                  builder.getStringAttr(label))
        .getOutput();
  }
  return value;
}

/// Creates a 2D identity matrix of the specified type.
static Value createIdentityMatrix(OpBuilder &builder, Location loc,
                                  RankedTensorType matrixType) {
  assert(matrixType.getRank() == 2 && "Expected 2D tensor type");
  assert(matrixType.getShape()[0] == matrixType.getShape()[1] &&
         "Expected square matrix type");

  int64_t size = matrixType.getShape()[0];
  auto elemType = matrixType.getElementType();

  SmallVector<Attribute> values;
  values.reserve(size * size);
  for (int64_t i = 0; i < size; ++i) {
    for (int64_t j = 0; j < size; ++j) {
      double val = (i == j) ? 1.0 : 0.0;
      values.push_back(builder.getFloatAttr(elemType, val));
    }
  }

  auto denseAttr = DenseElementsAttr::get(matrixType, values);
  return builder.create<arith::ConstantOp>(loc, matrixType, denseAttr);
}

/// Creates a permutation matrix of size n x n.
static Value createPermutationMatrix(OpBuilder &builder, Location loc,
                                     RankedTensorType matrixType) {
  assert(matrixType.getRank() == 2 && "Expected 2D tensor type");
  assert(matrixType.getShape()[0] == matrixType.getShape()[1] &&
         "Expected square matrix type");

  int64_t size = matrixType.getShape()[0];
  auto elemType = matrixType.getElementType();

  SmallVector<Attribute> values;
  values.reserve(size * size);
  for (int64_t i = 0; i < size; ++i) {
    for (int64_t j = 0; j < size; ++j) {
      double val = (j == size - 1 - i) ? 1.0 : 0.0;
      values.push_back(builder.getFloatAttr(elemType, val));
    }
  }

  auto denseAttr = DenseElementsAttr::get(matrixType, values);
  return builder.create<arith::ConstantOp>(loc, matrixType, denseAttr);
}

/// Computes A[::-1, ::-1] using permutation matrix through P @ A @ P.
static Value reverseRowsAndColumns(OpBuilder &builder, Location loc,
                                   Value matrix) {
  auto matrixType = cast<RankedTensorType>(matrix.getType());
  auto P = createPermutationMatrix(builder, loc, matrixType);

  // PA = P @ A
  auto PA = builder.create<impulse::DotOp>(
      loc, matrixType, P, matrix, builder.getDenseI64ArrayAttr({}),
      builder.getDenseI64ArrayAttr({}), builder.getDenseI64ArrayAttr({1}),
      builder.getDenseI64ArrayAttr({0}));

  // PAP = PA @ P
  return builder.create<impulse::DotOp>(
      loc, matrixType, PA, P, builder.getDenseI64ArrayAttr({}),
      builder.getDenseI64ArrayAttr({}), builder.getDenseI64ArrayAttr({1}),
      builder.getDenseI64ArrayAttr({0}));
}

Value impulse::applyInverseMassMatrix(OpBuilder &builder, Location loc,
                                      Value invMass, Value momentum,
                                      RankedTensorType positionType) {
  if (!invMass) {
    return momentum;
  }

  auto invMassType = cast<RankedTensorType>(invMass.getType());

  if (invMassType.getRank() == 1) {
    // Diagonal: element-wise
    auto diagMass =
        builder.create<impulse::ReshapeOp>(loc, positionType, invMass);
    return builder.create<arith::MulFOp>(loc, diagMass, momentum);
  } else if (invMassType.getRank() == 2) {
    // Dense: v = invMass @ p
    return builder.create<impulse::DotOp>(
        loc, positionType, momentum, invMass,
        builder.getDenseI64ArrayAttr({}), builder.getDenseI64ArrayAttr({}),
        builder.getDenseI64ArrayAttr({1}), builder.getDenseI64ArrayAttr({0}));
  }

  emitError(loc, "ProbProg: Provided invMass must have rank 1 or 2, got rank " +
                     std::to_string(invMassType.getRank()));
  return nullptr;
}

Value impulse::computeKineticEnergy(OpBuilder &builder, Location loc,
                                    Value momentum, Value invMass,
                                    RankedTensorType positionType) {
  auto elemType = positionType.getElementType();
  auto scalarType = RankedTensorType::get({}, elemType);

  auto halfConst = builder.create<arith::ConstantOp>(
      loc, scalarType,
      DenseElementsAttr::get(scalarType, builder.getFloatAttr(elemType, 0.5)));

  // v = M^-1 @ p
  auto v =
      applyInverseMassMatrix(builder, loc, invMass, momentum, positionType);

  // K = 0.5 * p^T @ v
  // For 2D tensors [1, N], contract over both dimensions to get scalar
  auto pDotV = builder.create<impulse::DotOp>(
      loc, scalarType, momentum, v, builder.getDenseI64ArrayAttr({}),
      builder.getDenseI64ArrayAttr({}), builder.getDenseI64ArrayAttr({0, 1}),
      builder.getDenseI64ArrayAttr({0, 1}));

  return builder.create<arith::MulFOp>(loc, halfConst, pDotV);
}

Value impulse::computeMassMatrixSqrt(OpBuilder &builder, Location loc,
                                     Value invMass,
                                     RankedTensorType positionType) {
  if (!invMass) {
    return Value();
  }

  auto invMassType = cast<RankedTensorType>(invMass.getType());
  auto elemType = invMassType.getElementType();

  if (invMassType.getRank() == 1) {
    // Diagonal: mass_matrix_sqrt = 1/sqrt(invMass)
    auto sqrtInvMass = builder.create<math::SqrtOp>(loc, invMass);
    auto onesVector = builder.create<arith::ConstantOp>(
        loc, invMassType,
        DenseElementsAttr::get(invMassType,
                               builder.getFloatAttr(elemType, 1.0)));
    return builder.create<arith::DivFOp>(loc, onesVector, sqrtInvMass);
  } else {
    // Dense: mass_matrix_sqrt = M^{1/2}
    // TODO: improve
    // Reference:
    // https://github.com/pyro-ppl/numpyro/blob/6a9cb9a530fe53897edb6c472368e58965b034e4/numpyro/infer/hmc_util.py#L499
    auto reversedInvMass = reverseRowsAndColumns(builder, loc, invMass);
    auto L_reversed =
        builder.create<impulse::CholeskyOp>(loc, invMassType, reversedInvMass,
                                    /*lower=*/builder.getBoolAttr(true));
    auto massMatrixSqrtInvT = reverseRowsAndColumns(builder, loc, L_reversed);
    auto identityMatrix = createIdentityMatrix(builder, loc, invMassType);
    auto massMatrixSqrt = builder.create<impulse::TriangularSolveOp>(
        loc, invMassType, massMatrixSqrtInvT, identityMatrix,
        /*left_side=*/builder.getBoolAttr(true),
        /*lower=*/builder.getBoolAttr(false),
        /*unit_diagonal=*/builder.getBoolAttr(false),
        /*transpose_a=*/
        impulse::TransposeAttr::get(builder.getContext(),
                                    impulse::Transpose::TRANSPOSE));

    return massMatrixSqrt;
  }
}

std::pair<Value, Value>
impulse::sampleMomentum(OpBuilder &builder, Location loc, Value rng,
                        Value invMass, Value massMatrixSqrt,
                        RankedTensorType positionType, bool debugDump) {
  auto elemType = positionType.getElementType();
  auto scalarType = RankedTensorType::get({}, elemType);

  auto zeroConst = builder.create<arith::ConstantOp>(
      loc, scalarType,
      DenseElementsAttr::get(scalarType, builder.getFloatAttr(elemType, 0.0)));
  auto oneConst = builder.create<arith::ConstantOp>(
      loc, scalarType,
      DenseElementsAttr::get(scalarType, builder.getFloatAttr(elemType, 1.0)));

  rng = conditionalDump(builder, loc, rng, "sampleMomentum: input rng state",
                        debugDump);

  // Sample eps ~ N(0, I)
  auto randomOp = builder.create<impulse::RandomOp>(
      loc, TypeRange{rng.getType(), positionType}, rng, zeroConst,
      oneConst,
      impulse::RngDistributionAttr::get(builder.getContext(),
                                        impulse::RngDistribution::NORMAL));

  auto rngOut = randomOp.getOutputRngState();
  auto eps = randomOp.getResult();

  if (!massMatrixSqrt) {
    return {eps, rngOut};
  }

  auto massMatrixSqrtType = cast<RankedTensorType>(massMatrixSqrt.getType());

  if (massMatrixSqrtType.getRank() == 1) {
    // Diagonal: p = massMatrixSqrt * eps (element-wise)
    auto diagSqrt =
        builder.create<impulse::ReshapeOp>(loc, positionType, massMatrixSqrt);
    auto p = builder.create<arith::MulFOp>(loc, diagSqrt, eps);
    return {p, rngOut};
  } else {
    // Dense: p = massMatrixSqrt @ eps
    auto p = builder.create<impulse::DotOp>(
        loc, positionType, eps, massMatrixSqrt,
        builder.getDenseI64ArrayAttr({}), builder.getDenseI64ArrayAttr({}),
        builder.getDenseI64ArrayAttr({1}), builder.getDenseI64ArrayAttr({1}));
    return {p, rngOut};
  }
}

static Value scatterPositionToTrace(OpBuilder &builder, Location loc,
                                    Value position2d, Value fullTrace,
                                    const HMCContext &ctx) {
  auto elemType =
      cast<RankedTensorType>(ctx.stepSize.getType()).getElementType();
  auto traceType = RankedTensorType::get({1, ctx.getFullTraceSize()}, elemType);
  auto i64TensorType = RankedTensorType::get({}, builder.getI64Type());
  auto c0 = builder.create<arith::ConstantOp>(
      loc, i64TensorType,
      DenseElementsAttr::get(i64TensorType, builder.getI64IntegerAttr(0)));

  Value result = fullTrace;
  for (const auto &info : ctx.supports) {
    auto sliceType = RankedTensorType::get({1, info.size}, elemType);
    auto posOffset = builder.create<arith::ConstantOp>(
        loc, i64TensorType,
        DenseElementsAttr::get(i64TensorType,
                               builder.getI64IntegerAttr(info.offset)));
    SmallVector<Value> extractIndices{c0, posOffset};
    auto slice = builder.create<impulse::DynamicSliceOp>(
        loc, sliceType, position2d, extractIndices,
        builder.getDenseI64ArrayAttr({1, info.size}));

    auto traceOffset = builder.create<arith::ConstantOp>(
        loc, i64TensorType,
        DenseElementsAttr::get(i64TensorType,
                               builder.getI64IntegerAttr(info.traceOffset)));
    SmallVector<Value> updateIndices{c0, traceOffset};
    result = builder.create<impulse::DynamicUpdateSliceOp>(
        loc, traceType, result, slice, updateIndices);
  }
  return result;
}

static Value gatherPositionFromTrace(OpBuilder &builder, Location loc,
                                     Value fullTrace, const HMCContext &ctx) {
  auto elemType =
      cast<RankedTensorType>(ctx.stepSize.getType()).getElementType();
  auto positionType2d = RankedTensorType::get({1, ctx.positionSize}, elemType);
  auto i64TensorType = RankedTensorType::get({}, builder.getI64Type());
  auto c0 = builder.create<arith::ConstantOp>(
      loc, i64TensorType,
      DenseElementsAttr::get(i64TensorType, builder.getI64IntegerAttr(0)));

  auto zeroConst = builder.create<arith::ConstantOp>(
      loc, positionType2d,
      DenseElementsAttr::get(positionType2d,
                             builder.getFloatAttr(elemType, 0.0)));
  Value result = zeroConst;
  for (const auto &info : ctx.supports) {
    auto sliceType = RankedTensorType::get({1, info.size}, elemType);
    auto traceOffset = builder.create<arith::ConstantOp>(
        loc, i64TensorType,
        DenseElementsAttr::get(i64TensorType,
                               builder.getI64IntegerAttr(info.traceOffset)));
    SmallVector<Value> extractIndices{c0, traceOffset};
    auto slice = builder.create<impulse::DynamicSliceOp>(
        loc, sliceType, fullTrace, extractIndices,
        builder.getDenseI64ArrayAttr({1, info.size}));

    auto posOffset = builder.create<arith::ConstantOp>(
        loc, i64TensorType,
        DenseElementsAttr::get(i64TensorType,
                               builder.getI64IntegerAttr(info.offset)));
    SmallVector<Value> updateIndices{c0, posOffset};
    result = builder.create<impulse::DynamicUpdateSliceOp>(
        loc, positionType2d, result, slice, updateIndices);
  }
  return result;
}

GradientResult impulse::computePotentialAndGradient(OpBuilder &builder,
                                                    Location loc,
                                                    Value position, Value rng,
                                                    const HMCContext &ctx) {
  auto positionType = ctx.getPositionType();
  auto scalarType = ctx.getScalarType();
  auto elemType = ctx.getElementType();

  auto gradSeed = builder.create<arith::ConstantOp>(
      loc, scalarType,
      DenseElementsAttr::get(scalarType, builder.getFloatAttr(elemType, 1.0)));

  bool isCustomLogpdf = ctx.hasCustomLogpdf();
  auto flatType = RankedTensorType::get({ctx.positionSize}, elemType);
  Value autodiffPosition = position;
  auto autodiffPositionType = positionType;
  auto autodiffGradType = positionType;
  if (isCustomLogpdf) {
    autodiffPosition =
        builder.create<impulse::ReshapeOp>(loc, flatType, position);
    autodiffPositionType = flatType;
    autodiffGradType = flatType;
  }

  SmallVector<Value> autodiffInputs{autodiffPosition, gradSeed};
  SmallVector<NamedAttribute> adAttrs{
      builder.getNamedAttr(
          "activity",
          builder.getArrayAttr({enzyme::ActivityAttr::get(
              builder.getContext(), enzyme::Activity::enzyme_active)})),
      builder.getNamedAttr(
          "ret_activity",
          builder.getArrayAttr(
              {enzyme::ActivityAttr::get(builder.getContext(),
                                         enzyme::Activity::enzyme_active),
               enzyme::ActivityAttr::get(builder.getContext(),
                                         enzyme::Activity::enzyme_const)})),
  };
  if (ctx.autodiffAttrs) {
    for (auto attr : ctx.autodiffAttrs)
      adAttrs.push_back(attr);
  }
  auto autodiffOp = builder.create<enzyme::AutoDiffRegionOp>(
      loc, TypeRange{scalarType, rng.getType(), autodiffGradType},
      autodiffInputs, adAttrs);

  Block *autodiffBlock = builder.createBlock(&autodiffOp.getBody());
  autodiffBlock->addArgument(autodiffPositionType, loc);

  builder.setInsertionPointToStart(autodiffBlock);
  Value qArg = autodiffBlock->getArgument(0);

  if (isCustomLogpdf) {
    SmallVector<Value> callArgs;
    callArgs.push_back(qArg);
    callArgs.append(ctx.fnInputs.begin(), ctx.fnInputs.end());
    auto callOp = builder.create<func::CallOp>(loc, ctx.logpdfFn,
                                       TypeRange{scalarType}, callArgs);
    Value U = builder.create<arith::NegFOp>(loc, callOp.getResult(0));
    // TODO(#2695): handle hybrid case
    builder.create<enzyme::YieldOp>(loc, ValueRange{U, rng});
  } else {
    auto traceType =
        RankedTensorType::get({1, ctx.getFullTraceSize()}, elemType);

    Value q_constrained = constrainPosition(builder, loc, qArg, ctx.supports);
    Value fullTrace = scatterPositionToTrace(builder, loc, q_constrained,
                                             ctx.originalTrace, ctx);

    SmallVector<Value> generateInputs;
    generateInputs.push_back(rng);
    generateInputs.append(ctx.fnInputs.begin(), ctx.fnInputs.end());

    SmallVector<Type> generateResultTypes;
    generateResultTypes.push_back(traceType);
    generateResultTypes.push_back(scalarType);
    generateResultTypes.append(ctx.fnResultTypes.begin(),
                               ctx.fnResultTypes.end());

    auto generateOp = builder.create<impulse::GenerateOp>(
        loc, generateResultTypes, ctx.fn, generateInputs, fullTrace,
        ctx.allAddresses, ctx.allAddresses, builder.getStringAttr(""));

    Value negWeight =
        builder.create<arith::NegFOp>(loc, generateOp.getWeight());
    Value jacobianCorrection =
        computeTotalJacobianCorrection(builder, loc, qArg, ctx.supports);
    Value U =
        builder.create<arith::SubFOp>(loc, negWeight, jacobianCorrection);

    SmallVector<Value> yieldValues{U, generateOp.getResult(2)};
    builder.create<enzyme::YieldOp>(loc, yieldValues);
  }

  builder.setInsertionPointAfter(autodiffOp);

  Value grad = autodiffOp.getResult(2);
  if (isCustomLogpdf) {
    grad = builder.create<impulse::ReshapeOp>(loc, positionType, grad);
  }

  return {
      autodiffOp.getResult(0), // U
      grad,                    // grad
      autodiffOp.getResult(1)  // rng
  };
}

IntegrationResult impulse::computeIntegrationStep(OpBuilder &builder,
                                                  Location loc,
                                                  const IntegratorState &leaf,
                                                  Value rng, Value direction,
                                                  const HMCContext &ctx) {
  auto positionType = ctx.getPositionType();
  auto scalarType = ctx.getScalarType();
  auto elemType = ctx.getElementType();

  auto negStepSize = builder.create<arith::NegFOp>(loc, ctx.stepSize);
  Value signedStepSize = builder.create<impulse::SelectOp>(
      loc, scalarType, direction, ctx.stepSize, negStepSize);

  auto halfConst = builder.create<arith::ConstantOp>(
      loc, scalarType,
      DenseElementsAttr::get(scalarType, builder.getFloatAttr(elemType, 0.5)));

  ArrayRef<int64_t> shape = positionType.getShape();
  auto stepSizeBroadcast =
      builder.create<BroadcastOp>(loc, positionType, signedStepSize,
                          builder.getDenseI64ArrayAttr(shape));
  auto halfStep =
      builder.create<arith::MulFOp>(loc, halfConst, signedStepSize);
  auto halfStepBroadcast =
      builder.create<BroadcastOp>(loc, positionType, halfStep,
                          builder.getDenseI64ArrayAttr(shape));

  // 1. Half step momentum: p_half = p - 0.5 * eps * grad
  auto deltaP1 =
      builder.create<arith::MulFOp>(loc, halfStepBroadcast, leaf.grad);
  Value pHalf = builder.create<arith::SubFOp>(loc, leaf.p, deltaP1);

  // 2. Full step position: q_new = q + eps * M^-1 * p_half
  Value v =
      applyInverseMassMatrix(builder, loc, ctx.invMass, pHalf, positionType);
  auto deltaQ = builder.create<arith::MulFOp>(loc, stepSizeBroadcast, v);
  Value qNew = builder.create<arith::AddFOp>(loc, leaf.q, deltaQ);

  // 3. Compute gradient at new position
  auto gradResult = computePotentialAndGradient(builder, loc, qNew, rng, ctx);

  // 4. Final half step momentum: p_new = p_half - 0.5 * eps * grad_new
  auto deltaP2 =
      builder.create<arith::MulFOp>(loc, halfStepBroadcast, gradResult.grad);
  Value pNew = builder.create<arith::SubFOp>(loc, pHalf, deltaP2);

  return {qNew, pNew, gradResult.grad, gradResult.U, gradResult.rng};
}

Value impulse::checkTurning(OpBuilder &builder, Location loc, Value pLeft,
                            Value pRight, Value pSum, const NUTSContext &ctx) {
  auto positionType = ctx.getPositionType();
  auto scalarType = ctx.getScalarType();
  auto elemType = ctx.getElementType();

  auto zeroConst = builder.create<arith::ConstantOp>(
      loc, scalarType,
      DenseElementsAttr::get(scalarType, builder.getFloatAttr(elemType, 0.0)));
  auto halfConst = builder.create<arith::ConstantOp>(
      loc, scalarType,
      DenseElementsAttr::get(scalarType, builder.getFloatAttr(elemType, 0.5)));

  Value vLeft =
      applyInverseMassMatrix(builder, loc, ctx.invMass, pLeft, positionType);
  Value vRight =
      applyInverseMassMatrix(builder, loc, ctx.invMass, pRight, positionType);

  // p_sum_centered = p_sum - (p_left + p_right) / 2
  auto halfBroadcast = builder.create<BroadcastOp>(
      loc, positionType, halfConst,
      builder.getDenseI64ArrayAttr(positionType.getShape()));

  auto pLeftPlusPRight = builder.create<arith::AddFOp>(loc, pLeft, pRight);
  auto halfSum =
      builder.create<arith::MulFOp>(loc, halfBroadcast, pLeftPlusPRight);
  Value pSumCentered = builder.create<arith::SubFOp>(loc, pSum, halfSum);

  auto leftAngle = builder.create<impulse::DotOp>(
      loc, scalarType, vLeft, pSumCentered,
      builder.getDenseI64ArrayAttr({}), builder.getDenseI64ArrayAttr({}),
      builder.getDenseI64ArrayAttr({0, 1}),
      builder.getDenseI64ArrayAttr({0, 1}));
  auto rightAngle = builder.create<impulse::DotOp>(
      loc, scalarType, vRight, pSumCentered,
      builder.getDenseI64ArrayAttr({}), builder.getDenseI64ArrayAttr({}),
      builder.getDenseI64ArrayAttr({0, 1}),
      builder.getDenseI64ArrayAttr({0, 1}));

  // turning = (left_angle <= 0) OR (right_angle <= 0)
  auto leftNeg = builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLE,
                                       leftAngle, zeroConst);
  auto rightNeg = builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLE,
                                        rightAngle, zeroConst);

  return builder.create<arith::OrIOp>(loc, leftNeg, rightNeg);
}

Value impulse::computeUniformTransitionProb(OpBuilder &builder, Location loc,
                                            Value currentWeight,
                                            Value newWeight) {
  Value weightDiff =
      builder.create<arith::SubFOp>(loc, newWeight, currentWeight);
  return builder.create<impulse::LogisticOp>(loc, weightDiff.getType(),
                                     weightDiff);
}

Value impulse::computeBiasedTransitionProb(OpBuilder &builder, Location loc,
                                           Value currentWeight, Value newWeight,
                                           Value turning, Value diverging) {
  auto resultType = cast<RankedTensorType>(currentWeight.getType());
  auto elemType = resultType.getElementType();

  auto zeroConst = builder.create<arith::ConstantOp>(
      loc, resultType,
      DenseElementsAttr::get(resultType, builder.getFloatAttr(elemType, 0.0)));
  auto oneConst = builder.create<arith::ConstantOp>(
      loc, resultType,
      DenseElementsAttr::get(resultType, builder.getFloatAttr(elemType, 1.0)));

  Value weightDiff =
      builder.create<arith::SubFOp>(loc, newWeight, currentWeight);
  Value expDiff = builder.create<math::ExpOp>(loc, weightDiff);
  Value clippedProb =
      builder.create<arith::MinimumFOp>(loc, oneConst, expDiff);
  Value turningOrDiverging =
      builder.create<arith::OrIOp>(loc, turning, diverging);
  return builder.create<arith::SelectOp>(loc, resultType, turningOrDiverging,
                                 zeroConst, clippedProb);
}

NUTSTreeState impulse::combineTrees(OpBuilder &builder, Location loc,
                                    const NUTSTreeState &tree,
                                    const NUTSTreeState &subTree,
                                    Value direction, Value rng, bool biased,
                                    const NUTSContext &ctx) {
  auto positionType = ctx.getPositionType();
  auto scalarType = ctx.getScalarType();
  auto elemType = ctx.getElementType();
  auto i64TensorType = RankedTensorType::get({}, builder.getI64Type());

  auto zeroConst = builder.create<arith::ConstantOp>(
      loc, scalarType,
      DenseElementsAttr::get(scalarType, builder.getFloatAttr(elemType, 0.0)));
  auto oneConst = builder.create<arith::ConstantOp>(
      loc, scalarType,
      DenseElementsAttr::get(scalarType, builder.getFloatAttr(elemType, 1.0)));

  auto qLeft = builder.create<impulse::SelectOp>(loc, positionType, direction,
                                         tree.q_left, subTree.q_left);
  auto pLeft = builder.create<impulse::SelectOp>(loc, positionType, direction,
                                         tree.p_left, subTree.p_left);
  auto gradLeft = builder.create<impulse::SelectOp>(
      loc, positionType, direction, tree.grad_left, subTree.grad_left);
  auto qRight = builder.create<impulse::SelectOp>(loc, positionType, direction,
                                          subTree.q_right, tree.q_right);
  auto pRight = builder.create<impulse::SelectOp>(loc, positionType, direction,
                                          subTree.p_right, tree.p_right);
  auto gradRight =
      builder.create<impulse::SelectOp>(loc, positionType, direction,
                                subTree.grad_right, tree.grad_right);

  auto combinedWeight = builder.create<impulse::LogAddExpOp>(
      loc, scalarType, tree.weight, subTree.weight);

  // Compute transition probability
  Value transitionProb;
  if (biased) {
    transitionProb =
        computeBiasedTransitionProb(builder, loc, tree.weight, subTree.weight,
                                    subTree.turning, subTree.diverging);
  } else {
    transitionProb =
        computeUniformTransitionProb(builder, loc, tree.weight, subTree.weight);
  }

  auto randomOp = builder.create<impulse::RandomOp>(
      loc, TypeRange{rng.getType(), scalarType}, rng, zeroConst,
      oneConst,
      impulse::RngDistributionAttr::get(builder.getContext(),
                                        impulse::RngDistribution::UNIFORM));
  auto rngOut = randomOp.getOutputRngState();
  auto uniformSample = randomOp.getResult();

  auto acceptNew = builder.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::OLT, uniformSample, transitionProb);

  auto qProposal =
      builder.create<impulse::SelectOp>(loc, positionType, acceptNew,
                                subTree.q_proposal, tree.q_proposal);
  auto gradProposal =
      builder.create<impulse::SelectOp>(loc, positionType, acceptNew,
                                subTree.grad_proposal, tree.grad_proposal);
  auto UProposal = builder.create<impulse::SelectOp>(
      loc, scalarType, acceptNew, subTree.U_proposal, tree.U_proposal);
  auto HProposal = builder.create<impulse::SelectOp>(
      loc, scalarType, acceptNew, subTree.H_proposal, tree.H_proposal);

  auto oneI64 = builder.create<arith::ConstantOp>(
      loc, i64TensorType,
      DenseElementsAttr::get(i64TensorType, builder.getI64IntegerAttr(1)));
  auto combinedDepth = builder.create<arith::AddIOp>(loc, tree.depth, oneI64);

  Value combinedTurning;
  if (biased) {
    auto turningCheck = checkTurning(
        builder, loc, pLeft, pRight,
        builder.create<arith::AddFOp>(loc, tree.p_sum, subTree.p_sum), ctx);
    combinedTurning =
        builder.create<arith::OrIOp>(loc, subTree.turning, turningCheck);
  } else {
    combinedTurning = tree.turning;
  }

  auto combinedDiverging =
      builder.create<arith::OrIOp>(loc, tree.diverging, subTree.diverging);
  auto sumAcceptProbs = builder.create<arith::AddFOp>(
      loc, tree.sum_accept_probs, subTree.sum_accept_probs);
  auto numProposals = builder.create<arith::AddIOp>(loc, tree.num_proposals,
                                            subTree.num_proposals);
  auto pSum = builder.create<arith::AddFOp>(loc, tree.p_sum, subTree.p_sum);

  return {.q_left = qLeft,
          .p_left = pLeft,
          .grad_left = gradLeft,
          .q_right = qRight,
          .p_right = pRight,
          .grad_right = gradRight,
          .q_proposal = qProposal,
          .grad_proposal = gradProposal,
          .U_proposal = UProposal,
          .H_proposal = HProposal,
          .depth = combinedDepth,
          .weight = combinedWeight,
          .turning = combinedTurning,
          .diverging = combinedDiverging,
          .sum_accept_probs = sumAcceptProbs,
          .num_proposals = numProposals,
          .p_sum = pSum,
          .rng = rngOut};
}

InitialHMCState impulse::InitHMC(OpBuilder &builder, Location loc, Value rng,
                                 const HMCContext &ctx, Value initialPosition,
                                 bool debugDump) {
  auto positionType = ctx.getPositionType();
  auto scalarType = ctx.getScalarType();
  auto elemType = ctx.getElementType();

  auto initSplit = builder.create<impulse::RandomSplitOp>(
      loc, TypeRange{rng.getType(), rng.getType()}, rng);
  auto kernelSplit = builder.create<impulse::RandomSplitOp>(
      loc, TypeRange{rng.getType(), rng.getType(), rng.getType()},
      initSplit.getResult(0));
  auto rngForSampleKernel = kernelSplit.getResult(0);
  auto rngForAutodiff = kernelSplit.getResult(1);

  Value q0;
  Value U0;

  if (ctx.hasCustomLogpdf()) {
    q0 = initialPosition;
    auto flatType = RankedTensorType::get({ctx.positionSize}, elemType);
    auto q0Flat = builder.create<impulse::ReshapeOp>(loc, flatType, q0);
    SmallVector<Value> callArgs;
    callArgs.push_back(q0Flat);
    callArgs.append(ctx.fnInputs.begin(), ctx.fnInputs.end());
    auto callOp = builder.create<func::CallOp>(loc, ctx.logpdfFn,
                                       TypeRange{scalarType}, callArgs);
    U0 = builder.create<arith::NegFOp>(loc, callOp.getResult(0));
  } else {
    auto fullTraceType =
        RankedTensorType::get({1, ctx.getFullTraceSize()}, elemType);

    // 1. Extract initial position vector (constrained)
    auto q0_constrained =
        gatherPositionFromTrace(builder, loc, ctx.originalTrace, ctx);

    // 2. Unconstrain to get position vector for HMC
    q0 = unconstrainPosition(builder, loc, q0_constrained, ctx.supports);

    // 3. Compute initial potential energy: U0 = -weight + correction
    Value fullTraceInit = scatterPositionToTrace(builder, loc, q0_constrained,
                                                 ctx.originalTrace, ctx);

    SmallVector<Value> generateInputsInit;
    generateInputsInit.push_back(rngForAutodiff);
    generateInputsInit.append(ctx.fnInputs.begin(), ctx.fnInputs.end());

    SmallVector<Type> generateResultTypesInit;
    generateResultTypesInit.push_back(fullTraceType);
    generateResultTypesInit.push_back(scalarType);
    generateResultTypesInit.append(ctx.fnResultTypes.begin(),
                                   ctx.fnResultTypes.end());

    auto generateOpInit = builder.create<impulse::GenerateOp>(
        loc, generateResultTypesInit, ctx.fn, generateInputsInit,
        fullTraceInit, ctx.allAddresses, ctx.allAddresses,
        builder.getStringAttr(""));

    auto weight0 = generateOpInit.getWeight();
    auto negWeight0 = builder.create<arith::NegFOp>(loc, weight0);
    auto jacobian0 =
        computeTotalJacobianCorrection(builder, loc, q0, ctx.supports);
    U0 = builder.create<arith::SubFOp>(loc, negWeight0, jacobian0);
  }

  // 4. Compute initial gradient at q0
  bool isCustomLogpdf = ctx.hasCustomLogpdf();
  auto flatType = RankedTensorType::get({ctx.positionSize}, elemType);
  Value autodiffQ0 = q0;
  auto autodiffQ0Type = positionType;
  auto autodiffGradType = positionType;
  if (isCustomLogpdf) {
    autodiffQ0 = builder.create<impulse::ReshapeOp>(loc, flatType, q0);
    autodiffQ0Type = flatType;
    autodiffGradType = flatType;
  }

  auto gradSeedInit = builder.create<arith::ConstantOp>(
      loc, scalarType,
      DenseElementsAttr::get(scalarType, builder.getFloatAttr(elemType, 1.0)));
  SmallVector<Value> autodiffInputs{autodiffQ0, gradSeedInit};
  SmallVector<NamedAttribute> adInitAttrs{
      builder.getNamedAttr(
          "activity",
          builder.getArrayAttr({enzyme::ActivityAttr::get(
              builder.getContext(), enzyme::Activity::enzyme_active)})),
      builder.getNamedAttr(
          "ret_activity",
          builder.getArrayAttr(
              {enzyme::ActivityAttr::get(builder.getContext(),
                                         enzyme::Activity::enzyme_active),
               enzyme::ActivityAttr::get(builder.getContext(),
                                         enzyme::Activity::enzyme_const)})),
  };
  if (ctx.autodiffAttrs) {
    for (auto attr : ctx.autodiffAttrs)
      adInitAttrs.push_back(attr);
  }
  auto autodiffInit = builder.create<enzyme::AutoDiffRegionOp>(
      loc,
      TypeRange{scalarType, rngForAutodiff.getType(), autodiffGradType},
      autodiffInputs, adInitAttrs);

  Block *autodiffInitBlock = builder.createBlock(&autodiffInit.getBody());
  autodiffInitBlock->addArgument(autodiffQ0Type, loc);

  builder.setInsertionPointToStart(autodiffInitBlock);
  auto q0Arg = autodiffInitBlock->getArgument(0);

  if (isCustomLogpdf) {
    SmallVector<Value> callArgs;
    callArgs.push_back(q0Arg);
    callArgs.append(ctx.fnInputs.begin(), ctx.fnInputs.end());
    auto callOpInner = builder.create<func::CallOp>(loc, ctx.logpdfFn,
                                            TypeRange{scalarType}, callArgs);
    auto U0_init =
        builder.create<arith::NegFOp>(loc, callOpInner.getResult(0));
    builder.create<enzyme::YieldOp>(loc, ValueRange{U0_init, rngForAutodiff});
  } else {
    auto fullTraceType =
        RankedTensorType::get({1, ctx.getFullTraceSize()}, elemType);
    auto q0Arg_constrained =
        constrainPosition(builder, loc, q0Arg, ctx.supports);
    Value fullTraceInner = scatterPositionToTrace(
        builder, loc, q0Arg_constrained, ctx.originalTrace, ctx);

    SmallVector<Value> generateInputsInner;
    generateInputsInner.push_back(rngForAutodiff);
    generateInputsInner.append(ctx.fnInputs.begin(), ctx.fnInputs.end());

    SmallVector<Type> generateResultTypesInner;
    generateResultTypesInner.push_back(fullTraceType);
    generateResultTypesInner.push_back(scalarType);
    generateResultTypesInner.append(ctx.fnResultTypes.begin(),
                                    ctx.fnResultTypes.end());

    auto generateOpInner = builder.create<impulse::GenerateOp>(
        loc, generateResultTypesInner, ctx.fn, generateInputsInner,
        fullTraceInner, ctx.allAddresses, ctx.allAddresses,
        builder.getStringAttr(""));

    auto negWeightInit =
        builder.create<arith::NegFOp>(loc, generateOpInner.getWeight());
    auto jacobianInit =
        computeTotalJacobianCorrection(builder, loc, q0Arg, ctx.supports);
    auto U0_init =
        builder.create<arith::SubFOp>(loc, negWeightInit, jacobianInit);

    SmallVector<Value> yieldValues{U0_init, generateOpInner.getResult(2)};
    builder.create<enzyme::YieldOp>(loc, yieldValues);
  }
  builder.setInsertionPointAfter(autodiffInit);

  Value grad0 = autodiffInit.getResult(2);
  if (isCustomLogpdf) {
    grad0 = builder.create<impulse::ReshapeOp>(loc, positionType, grad0);
  }

  return {q0, U0, grad0, rngForSampleKernel};
}

MCMCKernelResult impulse::SampleHMC(OpBuilder &builder, Location loc, Value q,
                                    Value grad, Value U, Value rng,
                                    const HMCContext &ctx, bool debugDump) {
  auto positionType = ctx.getPositionType();
  auto scalarType = ctx.getScalarType();
  auto elemType = ctx.getElementType();
  auto i64TensorType = RankedTensorType::get({}, builder.getI64Type());
  auto i1TensorType = RankedTensorType::get({}, builder.getI1Type());

  // 0. Compute num_steps and adjusted step_size
  // num_steps = ceil(trajectory_length / step_size)
  // adjusted_step_size = trajectory_length / num_steps
  auto trajDivStep =
      builder.create<arith::DivFOp>(loc, ctx.trajectoryLength, ctx.stepSize);
  auto numStepsF64 = builder.create<math::CeilOp>(loc, trajDivStep);
  auto numSteps =
      builder.create<arith::FPToSIOp>(loc, i64TensorType, numStepsF64);
  auto adjustedStepSize =
      builder.create<arith::DivFOp>(loc, ctx.trajectoryLength, numStepsF64);

  auto adjustedCtx = ctx.withStepSize(adjustedStepSize);

  // 1. Split RNG: [rngNext, rngMomentum, rngTransition]
  auto sampleKernelSplit = builder.create<impulse::RandomSplitOp>(
      loc, TypeRange{rng.getType(), rng.getType(), rng.getType()},
      rng);
  auto rngNext = sampleKernelSplit.getResult(0);
  auto rngMomentum = sampleKernelSplit.getResult(1);
  auto rngTransition = sampleKernelSplit.getResult(2);

  // 2. Sample fresh momentum p ~ N(0, M)
  Value rngForMomentum = rngMomentum;
  if (!ctx.hasCustomLogpdf()) {
    auto momSplit = builder.create<impulse::RandomSplitOp>(
        loc, TypeRange{rng.getType(), rng.getType()}, rngMomentum);
    rngForMomentum = momSplit.getResult(0);
  }
  auto [p0, rngAfterMomentum] =
      sampleMomentum(builder, loc, rngForMomentum, ctx.invMass,
                     ctx.massMatrixSqrt, positionType, debugDump);

  // 3. Compute K0 = 0.5 * p^T * M^-1 * p
  auto K0 = computeKineticEnergy(builder, loc, p0, ctx.invMass, positionType);

  // 4. Compute H0 = U + K0
  auto H0 = builder.create<arith::AddFOp>(loc, U, K0);

  // 5. Leapfrog integration loop
  auto direction = builder.create<arith::ConstantOp>(
      loc, i1TensorType,
      DenseElementsAttr::get(i1TensorType, builder.getBoolAttr(true)));

  auto c0 = builder.create<arith::ConstantOp>(
      loc, i64TensorType,
      DenseElementsAttr::get(i64TensorType, builder.getI64IntegerAttr(0)));
  auto c1 = builder.create<arith::ConstantOp>(
      loc, i64TensorType,
      DenseElementsAttr::get(i64TensorType, builder.getI64IntegerAttr(1)));

  // Loop carries: [q, p, grad, U, rng]
  SmallVector<Type> loopResultTypes = {positionType, positionType, positionType,
                                       scalarType, rngTransition.getType()};
  auto forLoopOp =
      builder.create<impulse::ForOp>(loc, loopResultTypes, c0, numSteps, c1,
                             ValueRange{q, p0, grad, U, rngTransition});

  Block *loopBody = builder.createBlock(&forLoopOp.getRegion());
  loopBody->addArgument(i64TensorType, loc);           // iv
  loopBody->addArgument(positionType, loc);            // q
  loopBody->addArgument(positionType, loc);            // p
  loopBody->addArgument(positionType, loc);            // gradient
  loopBody->addArgument(scalarType, loc);              // U
  loopBody->addArgument(rngTransition.getType(), loc); // rng

  builder.setInsertionPointToStart(loopBody);
  auto qLoop = loopBody->getArgument(1);
  auto pLoop = loopBody->getArgument(2);
  auto gradLoop = loopBody->getArgument(3);
  auto rngLoop = loopBody->getArgument(5);

  IntegratorState leaf = {qLoop, pLoop, gradLoop};
  auto step = computeIntegrationStep(builder, loc, leaf, rngLoop, direction,
                                     adjustedCtx);

  // Yield [q, p, grad, U, rng]
  builder.create<impulse::YieldOp>(
      loc, ValueRange{step.q, step.p, step.grad, step.U, step.rng});

  builder.setInsertionPointAfter(forLoopOp);
  auto qProposal = forLoopOp.getResult(0);
  auto pProposal = forLoopOp.getResult(1);
  auto gradProposal = forLoopOp.getResult(2);
  auto UProposal = forLoopOp.getResult(3);
  auto rngAfterLeapfrog = forLoopOp.getResult(4);

  // 6. Compute K1, H1 for proposal
  auto K1 =
      computeKineticEnergy(builder, loc, pProposal, ctx.invMass, positionType);
  auto H1 = builder.create<arith::AddFOp>(loc, UProposal, K1);

  // 7. MH accept/reject
  auto zeroConst = builder.create<arith::ConstantOp>(
      loc, scalarType,
      DenseElementsAttr::get(scalarType, builder.getFloatAttr(elemType, 0.0)));
  auto oneConst = builder.create<arith::ConstantOp>(
      loc, scalarType,
      DenseElementsAttr::get(scalarType, builder.getFloatAttr(elemType, 1.0)));

  // α = min(1, exp(H0 - H1))
  auto dH = builder.create<arith::SubFOp>(loc, H0, H1);
  auto expDH = builder.create<math::ExpOp>(loc, dH);
  auto accProb = builder.create<arith::MinimumFOp>(loc, oneConst, expDH);

  // u ~ Uniform(0, 1)
  auto randomOp = builder.create<impulse::RandomOp>(
      loc, TypeRange{rngAfterLeapfrog.getType(), scalarType},
      rngAfterLeapfrog, zeroConst, oneConst,
      impulse::RngDistributionAttr::get(builder.getContext(),
                                        impulse::RngDistribution::UNIFORM));
  auto randUniform = randomOp.getResult();

  // accepted = u < α
  auto acceptedTensor = builder.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::OLT, randUniform, accProb);

  // 8. Select between original and proposal
  auto qFinal = builder.create<impulse::SelectOp>(loc, positionType,
                                          acceptedTensor, qProposal, q);
  auto gradFinal = builder.create<impulse::SelectOp>(
      loc, positionType, acceptedTensor, gradProposal, grad);
  auto UFinal = builder.create<impulse::SelectOp>(loc, scalarType,
                                          acceptedTensor, UProposal, U);

  return {qFinal, gradFinal, UFinal, acceptedTensor, accProb, rngNext};
}

MCMCKernelResult impulse::SampleNUTS(OpBuilder &builder, Location loc, Value q,
                                     Value grad, Value U, Value rng,
                                     const NUTSContext &ctx, bool debugDump) {
  auto positionType = ctx.getPositionType();
  auto scalarType = ctx.getScalarType();
  auto elemType = ctx.getElementType();
  auto i64TensorType = RankedTensorType::get({}, builder.getI64Type());
  auto i1TensorType = RankedTensorType::get({}, builder.getI1Type());

  // 1. Split RNG: [rngNext, rngMomentum, rngTree]
  auto sampleKernelSplit = builder.create<impulse::RandomSplitOp>(
      loc, TypeRange{rng.getType(), rng.getType(), rng.getType()},
      rng);
  auto rngNext = sampleKernelSplit.getResult(0);
  auto rngMomentum = sampleKernelSplit.getResult(1);
  auto rngTree = sampleKernelSplit.getResult(2);

  // 2. Sample fresh momentum p ~ N(0, M)
  Value rngForMomentum = rngMomentum;
  if (!ctx.hasCustomLogpdf()) {
    auto momSplit = builder.create<impulse::RandomSplitOp>(
        loc, TypeRange{rng.getType(), rng.getType()}, rngMomentum);
    rngForMomentum = momSplit.getResult(0);
  }
  auto [p0, rngAfterMomentum] =
      sampleMomentum(builder, loc, rngForMomentum, ctx.invMass,
                     ctx.massMatrixSqrt, positionType, debugDump);

  // 3. Compute K0 = 0.5 * p^T * M^-1 * p
  auto K0 = computeKineticEnergy(builder, loc, p0, ctx.invMass, positionType);

  // 4. Compute H0 = U + K0
  auto H0 = builder.create<arith::AddFOp>(loc, U, K0);

  // 5. Initialize NUTS tree state
  auto iterCtx = ctx.withH0(H0);

  auto zeroI64 = builder.create<arith::ConstantOp>(
      loc, i64TensorType,
      DenseElementsAttr::get(i64TensorType, builder.getI64IntegerAttr(0)));
  auto falseConst = builder.create<arith::ConstantOp>(
      loc, i1TensorType,
      DenseElementsAttr::get(i1TensorType, builder.getBoolAttr(false)));
  auto zeroConst = builder.create<arith::ConstantOp>(
      loc, scalarType,
      DenseElementsAttr::get(scalarType, builder.getFloatAttr(elemType, 0.0)));

  NUTSTreeState initialTree = {.q_left = q,
                               .p_left = p0,
                               .grad_left = grad,
                               .q_right = q,
                               .p_right = p0,
                               .grad_right = grad,
                               .q_proposal = q,
                               .grad_proposal = grad,
                               .U_proposal = U,
                               .H_proposal = H0,
                               .depth = zeroI64,
                               .weight = zeroConst,
                               .turning = falseConst,
                               .diverging = falseConst,
                               .sum_accept_probs = zeroConst,
                               .num_proposals = zeroI64,
                               .p_sum = p0,
                               .rng = rngTree};

  // 6. Build NUTS tree
  auto finalTree = buildTree(builder, loc, initialTree, iterCtx, debugDump);

  // 7. NUTS always accepts the proposal (implicit acceptance in the tree)
  auto trueConst = builder.create<arith::ConstantOp>(
      loc, i1TensorType,
      DenseElementsAttr::get(i1TensorType, builder.getBoolAttr(true)));

  // 8. Compute mean acceptance probability for step size adaptation
  auto oneI64 = builder.create<arith::ConstantOp>(
      loc, i64TensorType,
      DenseElementsAttr::get(i64TensorType, builder.getI64IntegerAttr(1)));
  auto numProposalsClamped =
      builder.create<arith::MaxSIOp>(loc, finalTree.num_proposals, oneI64);
  auto numProposalsFloat =
      builder.create<arith::SIToFPOp>(loc, scalarType, numProposalsClamped);
  auto meanAcceptProb = builder.create<arith::DivFOp>(
      loc, finalTree.sum_accept_probs, numProposalsFloat);

  return {finalTree.q_proposal, finalTree.grad_proposal,
          finalTree.U_proposal, trueConst,
          meanAcceptProb,       rngNext};
}

NUTSTreeState impulse::buildBaseTree(OpBuilder &builder, Location loc,
                                     const IntegratorState &leaf, Value rng,
                                     Value direction, const NUTSContext &ctx) {
  auto positionType = ctx.getPositionType();
  auto scalarType = ctx.getScalarType();
  auto elemType = ctx.getElementType();
  auto i64TensorType = RankedTensorType::get({}, builder.getI64Type());
  auto i1TensorType = RankedTensorType::get({}, builder.getI1Type());

  auto oneConst = builder.create<arith::ConstantOp>(
      loc, scalarType,
      DenseElementsAttr::get(scalarType, builder.getFloatAttr(elemType, 1.0)));
  auto zeroI64 = builder.create<arith::ConstantOp>(
      loc, i64TensorType,
      DenseElementsAttr::get(i64TensorType, builder.getI64IntegerAttr(0)));
  auto oneI64 = builder.create<arith::ConstantOp>(
      loc, i64TensorType,
      DenseElementsAttr::get(i64TensorType, builder.getI64IntegerAttr(1)));
  auto falseConst = builder.create<arith::ConstantOp>(
      loc, i1TensorType,
      DenseElementsAttr::get(i1TensorType, builder.getBoolAttr(false)));

  IntegrationResult leap =
      computeIntegrationStep(builder, loc, leaf, rng, direction, ctx);

  auto qNew = leap.q;
  auto pNew = leap.p;
  auto gradNew = leap.grad;
  auto UNew = leap.U;
  auto rngOut = leap.rng;

  auto KNew =
      computeKineticEnergy(builder, loc, pNew, ctx.invMass, positionType);
  auto HNew = builder.create<arith::AddFOp>(loc, UNew, KNew);
  Value deltaH = builder.create<arith::SubFOp>(loc, HNew, ctx.H0);

  // NaN check
  auto isNan = builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNE,
                                     deltaH, deltaH);
  auto infConst = builder.create<arith::ConstantOp>(
      loc, scalarType,
      DenseElementsAttr::get(
          scalarType, builder.getFloatAttr(
                          elemType, std::numeric_limits<double>::infinity())));
  deltaH = builder.create<arith::SelectOp>(loc, scalarType, isNan, infConst,
                                   deltaH);

  auto treeWeight = builder.create<arith::NegFOp>(loc, deltaH);

  // Check for divergence
  auto diverging = builder.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::OGT, deltaH, ctx.maxDeltaEnergy);

  auto negDeltaH = builder.create<arith::NegFOp>(loc, deltaH);
  auto expNegDelta = builder.create<math::ExpOp>(loc, negDeltaH);
  auto acceptProb =
      builder.create<arith::MinimumFOp>(loc, oneConst, expNegDelta);

  return {.q_left = qNew,
          .p_left = pNew,
          .grad_left = gradNew,
          .q_right = qNew,
          .p_right = pNew,
          .grad_right = gradNew,
          .q_proposal = qNew,
          .grad_proposal = gradNew,
          .U_proposal = UNew,
          .H_proposal = HNew,
          .depth = zeroI64,
          .weight = treeWeight,
          .turning = falseConst,
          .diverging = diverging,
          .sum_accept_probs = acceptProb,
          .num_proposals = oneI64,
          .p_sum = pNew,
          .rng = rngOut};
}

IntegratorState impulse::getLeafFromTree(OpBuilder &builder, Location loc,
                                         const NUTSTreeState &tree,
                                         Value direction,
                                         const NUTSContext &ctx) {
  auto positionType = ctx.getPositionType();

  auto leafQ = builder.create<impulse::SelectOp>(loc, positionType, direction,
                                         tree.q_right, tree.q_left);
  auto leafP = builder.create<impulse::SelectOp>(loc, positionType, direction,
                                         tree.p_right, tree.p_left);
  auto leafGrad = builder.create<impulse::SelectOp>(
      loc, positionType, direction, tree.grad_right, tree.grad_left);
  return {leafQ, leafP, leafGrad};
}

SubtreeBuildResult
impulse::buildIterativeSubtree(OpBuilder &builder, Location loc,
                               const NUTSTreeState &initialTree,
                               Value direction, Value pCkpts, Value pSumCkpts,
                               const NUTSContext &ctx, bool debugDump) {
  auto i1TensorType = RankedTensorType::get({}, builder.getI1Type());
  auto i64TensorType = RankedTensorType::get({}, builder.getI64Type());
  auto pCkptsType = cast<RankedTensorType>(pCkpts.getType());
  auto trueConst = builder.create<arith::ConstantOp>(
      loc, i1TensorType,
      DenseElementsAttr::get(i1TensorType, builder.getBoolAttr(true)));
  auto oneI64 = builder.create<arith::ConstantOp>(
      loc, i64TensorType,
      DenseElementsAttr::get(i64TensorType, builder.getI64IntegerAttr(1)));
  auto zeroI64 = builder.create<arith::ConstantOp>(
      loc, i64TensorType,
      DenseElementsAttr::get(i64TensorType, builder.getI64IntegerAttr(0)));

  // 2 ^ (initialTree.depth)
  auto maxNumProposals =
      builder.create<arith::ShLIOp>(loc, oneI64, initialTree.depth);

  SmallVector<Type> whileTypes = initialTree.getTypes();
  whileTypes.push_back(pCkptsType);
  whileTypes.push_back(pCkptsType);
  whileTypes.push_back(i64TensorType);

  SmallVector<Value> whileInitVals = initialTree.toValues();
  whileInitVals[15] = zeroI64; // zero `num_proposals`
  whileInitVals.push_back(pCkpts);
  whileInitVals.push_back(pSumCkpts);
  whileInitVals.push_back(zeroI64);

  auto whileOp =
      builder.create<impulse::WhileOp>(loc, whileTypes, whileInitVals);

  // Check: num_proposals < max_num_proposals && !turning && !diverging
  Block *condBlock = builder.createBlock(&whileOp.getConditionRegion());
  for (auto type : whileTypes)
    condBlock->addArgument(type, loc);

  builder.setInsertionPointToStart(condBlock);
  SmallVector<Value> condTreeArgs(condBlock->getArguments().begin(),
                                  condBlock->getArguments().begin() + 18);
  NUTSTreeState condTree = NUTSTreeState::fromValues(condTreeArgs);

  auto numProposalsCheck =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                            condTree.num_proposals, maxNumProposals);
  auto notTurning =
      builder.create<arith::XOrIOp>(loc, condTree.turning, trueConst);
  auto notDiverging =
      builder.create<arith::XOrIOp>(loc, condTree.diverging, trueConst);
  auto continueCond = builder.create<arith::AndIOp>(
      loc,
      builder.create<arith::AndIOp>(loc, numProposalsCheck, notTurning),
      notDiverging);

  // Yield continue condition
  builder.create<impulse::YieldOp>(loc, ValueRange{continueCond});

  Block *bodyBlock = builder.createBlock(&whileOp.getBodyRegion());
  for (auto type : whileTypes)
    bodyBlock->addArgument(type, loc);

  builder.setInsertionPointToStart(bodyBlock);

  SmallVector<Value> bodyTreeArgs(bodyBlock->getArguments().begin(),
                                  bodyBlock->getArguments().begin() + 18);
  NUTSTreeState bodyTree = NUTSTreeState::fromValues(bodyTreeArgs);
  auto bodyPCkpts = bodyBlock->getArgument(18);
  auto bodyPSumCkpts = bodyBlock->getArgument(19);
  auto bodyLeafIdx = bodyBlock->getArgument(20);

  // Extract leaf based on direction
  IntegratorState leaf =
      getLeafFromTree(builder, loc, bodyTree, direction, ctx);

  auto rngSplit2 = builder.create<impulse::RandomSplitOp>(
      loc, TypeRange{bodyTree.rng.getType(), bodyTree.rng.getType()},
      bodyTree.rng);
  auto rngNext = rngSplit2.getResult(0);
  auto rngCombine = rngSplit2.getResult(1);

  // Build base tree
  NUTSTreeState newLeaf =
      buildBaseTree(builder, loc, leaf, rngNext, direction, ctx);

  // First leaf check
  auto isFirstLeaf = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, bodyTree.num_proposals, zeroI64);

  SmallVector<Type> treeTypes = newLeaf.getTypes();
  auto ifOp = builder.create<impulse::IfOp>(loc, treeTypes, isFirstLeaf);
  {
    Block *trueBranch = builder.createBlock(&ifOp.getTrueBranch());
    builder.setInsertionPointToStart(trueBranch);
    builder.create<impulse::YieldOp>(loc, newLeaf.toValues());
  }
  {
    Block *falseBranch = builder.createBlock(&ifOp.getFalseBranch());
    builder.setInsertionPointToStart(falseBranch);
    NUTSTreeState combinedTree =
        combineTrees(builder, loc, bodyTree, newLeaf, direction, rngCombine,
                     /*biased=*/false, ctx);
    builder.create<impulse::YieldOp>(loc, combinedTree.toValues());
  }

  builder.setInsertionPointAfter(ifOp);
  NUTSTreeState updatedTree = NUTSTreeState::fromValues(
      SmallVector<Value>(ifOp.getResults().begin(), ifOp.getResults().end()));
  updatedTree.rng = rngNext;

  // Update and check iterative turning
  auto [ckptIdxMin, ckptIdxMax] =
      leafIdxToCheckpointIdxs(builder, loc, bodyLeafIdx);
  auto [updatedPCkpts, updatedPSumCkpts] = updateCheckpoints(
      builder, loc, bodyLeafIdx, ckptIdxMax, newLeaf.p_right, updatedTree.p_sum,
      bodyPCkpts, bodyPSumCkpts, ctx, debugDump);
  auto iterativeTurning = checkIterativeTurning(
      builder, loc, newLeaf.p_right, updatedTree.p_sum, updatedPCkpts,
      updatedPSumCkpts, ckptIdxMin, ckptIdxMax, ctx, debugDump);

  updatedTree.turning =
      builder.create<impulse::SelectOp>(loc, i1TensorType, isFirstLeaf,
                                newLeaf.turning, iterativeTurning);

  auto nextLeafIdx = builder.create<arith::AddIOp>(loc, bodyLeafIdx, oneI64);

  SmallVector<Value> yieldVals = updatedTree.toValues();
  yieldVals.push_back(updatedPCkpts);
  yieldVals.push_back(updatedPSumCkpts);
  yieldVals.push_back(nextLeafIdx);
  builder.create<impulse::YieldOp>(loc, yieldVals);

  builder.setInsertionPointAfter(whileOp);

  SmallVector<Value> resultTreeArgs(whileOp.getResults().begin(),
                                    whileOp.getResults().begin() + 18);
  NUTSTreeState resultTree = NUTSTreeState::fromValues(resultTreeArgs);
  auto resultPCkpts = whileOp.getResult(18);
  auto resultPSumCkpts = whileOp.getResult(19);

  // `combineTrees` increments depth at each leaf building step; we need to
  // restore to target depth here
  resultTree.depth = initialTree.depth;

  return {resultTree, resultPCkpts, resultPSumCkpts};
}

SubtreeBuildResult impulse::doubleTree(OpBuilder &builder, Location loc,
                                       const NUTSTreeState &tree,
                                       Value direction, Value pCkpts,
                                       Value pSumCkpts, const NUTSContext &ctx,
                                       bool debugDump) {
  auto rngSplit2 = builder.create<impulse::RandomSplitOp>(
      loc, TypeRange{tree.rng.getType(), tree.rng.getType()},
      tree.rng);
  auto rngSubtree = rngSplit2.getResult(0);
  auto rngTransition = rngSplit2.getResult(1);

  NUTSTreeState subTreeInit = tree;
  subTreeInit.rng = rngSubtree;
  auto subtreeResult = buildIterativeSubtree(
      builder, loc, subTreeInit, direction, pCkpts, pSumCkpts, ctx, debugDump);

  // Tree combine using *biased* transition kernel
  NUTSTreeState combinedTree =
      combineTrees(builder, loc, tree, subtreeResult.tree, direction,
                   rngTransition, /*biased=*/true, ctx);

  return {combinedTree, subtreeResult.pCkpts, subtreeResult.pSumCkpts};
}

NUTSTreeState impulse::buildTree(OpBuilder &builder, Location loc,
                                 const NUTSTreeState &initialTree,
                                 const NUTSContext &ctx, bool debugDump) {
  auto elemType =
      cast<RankedTensorType>(ctx.stepSize.getType()).getElementType();
  auto F64TensorType = RankedTensorType::get({}, elemType);
  auto i64TensorType = RankedTensorType::get({}, builder.getI64Type());
  auto i1TensorType = RankedTensorType::get({}, builder.getI1Type());

  auto trueConst = builder.create<arith::ConstantOp>(
      loc, i1TensorType,
      DenseElementsAttr::get(i1TensorType, builder.getBoolAttr(true)));
  auto halfConst = builder.create<arith::ConstantOp>(
      loc, F64TensorType,
      DenseElementsAttr::get(F64TensorType, builder.getF64FloatAttr(0.5)));
  auto zeroConst = builder.create<arith::ConstantOp>(
      loc, F64TensorType,
      DenseElementsAttr::get(F64TensorType, builder.getF64FloatAttr(0.0)));
  auto oneConst = builder.create<arith::ConstantOp>(
      loc, F64TensorType,
      DenseElementsAttr::get(F64TensorType, builder.getF64FloatAttr(1.0)));

  auto maxTreeDepth = builder.create<arith::ConstantOp>(
      loc, i64TensorType,
      DenseElementsAttr::get(i64TensorType,
                             builder.getI64IntegerAttr(ctx.maxTreeDepth)));

  auto checkpointType =
      RankedTensorType::get({ctx.maxTreeDepth, ctx.positionSize}, elemType);

  SmallVector<Type> whileTypes = initialTree.getTypes();
  SmallVector<Value> whileInitVals = initialTree.toValues();

  auto whileOp =
      builder.create<impulse::WhileOp>(loc, whileTypes, whileInitVals);

  // Check: (depth < maxTreeDepth) && !turning && !diverging
  Block *condBlock = builder.createBlock(&whileOp.getConditionRegion());
  for (auto type : whileTypes)
    condBlock->addArgument(type, loc);

  builder.setInsertionPointToStart(condBlock);

  SmallVector<Value> condArgs(condBlock->getArguments().begin(),
                              condBlock->getArguments().end());
  NUTSTreeState condTree = NUTSTreeState::fromValues(condArgs);

  auto depthCheck = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, condTree.depth, maxTreeDepth);
  auto notTurning =
      builder.create<arith::XOrIOp>(loc, condTree.turning, trueConst);
  auto notDiverging =
      builder.create<arith::XOrIOp>(loc, condTree.diverging, trueConst);

  // Yield continue condition
  auto continueCond = builder.create<arith::AndIOp>(
      loc, builder.create<arith::AndIOp>(loc, depthCheck, notTurning),
      notDiverging);

  builder.create<impulse::YieldOp>(loc, ValueRange{continueCond});

  Block *bodyBlock = builder.createBlock(&whileOp.getBodyRegion());
  for (auto type : whileTypes)
    bodyBlock->addArgument(type, loc);

  builder.setInsertionPointToStart(bodyBlock);

  SmallVector<Value> bodyArgs(bodyBlock->getArguments().begin(),
                              bodyBlock->getArguments().end());
  NUTSTreeState bodyTree = NUTSTreeState::fromValues(bodyArgs);

  // Create fresh checkpoint tensors
  auto zeroCkpts = builder.create<arith::ConstantOp>(
      loc, checkpointType,
      DenseElementsAttr::get(checkpointType, builder.getF64FloatAttr(0.0)));
  Value bodyPCkpts = zeroCkpts;
  Value bodyPSumCkpts = zeroCkpts;

  auto rngSplit3 = builder.create<impulse::RandomSplitOp>(
      loc,
      TypeRange{bodyTree.rng.getType(), bodyTree.rng.getType(),
                bodyTree.rng.getType()},
      bodyTree.rng);
  auto rngNext = rngSplit3.getResult(0);
  auto rngDir = rngSplit3.getResult(1);
  auto rngDbl = rngSplit3.getResult(2);

  auto directionRandom = builder.create<impulse::RandomOp>(
      loc, TypeRange{rngDir.getType(), F64TensorType}, rngDir,
      zeroConst, oneConst,
      impulse::RngDistributionAttr::get(builder.getContext(),
                                        impulse::RngDistribution::UNIFORM));
  auto direction =
      builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT,
                            directionRandom.getResult(), halfConst);

  // Double the tree
  NUTSTreeState treeToDouble = bodyTree;
  treeToDouble.rng = rngDbl;
  auto doubleResult = doubleTree(builder, loc, treeToDouble, direction,
                                 bodyPCkpts, bodyPSumCkpts, ctx, debugDump);

  NUTSTreeState treeToYield = doubleResult.tree;
  treeToYield.rng = rngNext;
  builder.create<impulse::YieldOp>(loc, treeToYield.toValues());

  builder.setInsertionPointAfter(whileOp);

  SmallVector<Value> results(whileOp.getResults().begin(),
                             whileOp.getResults().end());
  return NUTSTreeState::fromValues(results);
}

std::pair<Value, Value> impulse::leafIdxToCheckpointIdxs(OpBuilder &builder,
                                                         Location loc,
                                                         Value leafIdx) {
  auto i64TensorType = cast<RankedTensorType>(leafIdx.getType());

  auto oneConst = builder.create<arith::ConstantOp>(
      loc, i64TensorType,
      DenseElementsAttr::get(i64TensorType, builder.getI64IntegerAttr(1)));

  // idx_max = popcount(leafIdx >> 1)
  auto shiftedIdx = builder.create<arith::ShRUIOp>(loc, leafIdx, oneConst);
  auto idxMax =
      builder.create<impulse::PopcountOp>(loc, i64TensorType, shiftedIdx);

  // num_subtrees = popcount((~leafIdx & (leafIdx + 1)) - 1)
  auto leafIdxPlusOne = builder.create<arith::AddIOp>(loc, leafIdx, oneConst);

  auto minusOneConst = builder.create<arith::ConstantOp>(
      loc, i64TensorType,
      DenseElementsAttr::get(i64TensorType, builder.getI64IntegerAttr(-1)));
  auto notLeafIdx = builder.create<arith::XOrIOp>(loc, leafIdx, minusOneConst);

  Value andResult =
      builder.create<arith::AndIOp>(loc, notLeafIdx, leafIdxPlusOne);
  Value andMinusOne = builder.create<arith::SubIOp>(loc, andResult, oneConst);
  Value numSubtrees =
      builder.create<impulse::PopcountOp>(loc, i64TensorType, andMinusOne);

  // idx_min = idx_max - num_subtrees + 1
  Value idxMaxMinusNumSubtrees =
      builder.create<arith::SubIOp>(loc, idxMax, numSubtrees);
  Value idxMin =
      builder.create<arith::AddIOp>(loc, idxMaxMinusNumSubtrees, oneConst);

  return {idxMin, idxMax};
}

Value impulse::checkIterativeTurning(OpBuilder &builder, Location loc, Value p,
                                     Value pSum, Value pCkpts, Value pSumCkpts,
                                     Value idxMin, Value idxMax,
                                     const NUTSContext &ctx, bool debugDump) {
  auto positionType = ctx.getPositionType();
  auto i64TensorType = RankedTensorType::get({}, builder.getI64Type());
  auto i1TensorType = RankedTensorType::get({}, builder.getI1Type());

  auto falseConst = builder.create<arith::ConstantOp>(
      loc, i1TensorType,
      DenseElementsAttr::get(i1TensorType, builder.getBoolAttr(false)));
  auto oneI64 = builder.create<arith::ConstantOp>(
      loc, i64TensorType,
      DenseElementsAttr::get(i64TensorType, builder.getI64IntegerAttr(1)));

  // Iterate from `idx_max` down to `idx_min`, check turning at each checkpoint
  SmallVector<Type> whileTypes = {i64TensorType, i1TensorType};
  SmallVector<Value> whileInitVals = {idxMax, falseConst};

  auto whileOp =
      builder.create<impulse::WhileOp>(loc, whileTypes, whileInitVals);
  Block *condBlock = builder.createBlock(&whileOp.getConditionRegion());
  condBlock->addArgument(i64TensorType, loc);
  condBlock->addArgument(i1TensorType, loc);
  builder.setInsertionPointToStart(condBlock);

  Value iCond = condBlock->getArgument(0);
  Value turningCond = condBlock->getArgument(1);

  auto iGeMin = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                      iCond, idxMin);
  auto trueConst = builder.create<arith::ConstantOp>(
      loc, i1TensorType,
      DenseElementsAttr::get(i1TensorType, builder.getBoolAttr(true)));
  auto notTurning = builder.create<arith::XOrIOp>(loc, turningCond, trueConst);
  auto continueLoop = builder.create<arith::AndIOp>(loc, iGeMin, notTurning);

  builder.create<impulse::YieldOp>(loc, ValueRange{continueLoop.getResult()});

  Block *bodyBlock = builder.createBlock(&whileOp.getBodyRegion());
  bodyBlock->addArgument(i64TensorType, loc);
  bodyBlock->addArgument(i1TensorType, loc);
  builder.setInsertionPointToStart(bodyBlock);
  Value iBody = bodyBlock->getArgument(0);

  auto zeroI64 = builder.create<arith::ConstantOp>(
      loc, i64TensorType,
      DenseElementsAttr::get(i64TensorType, builder.getI64IntegerAttr(0)));
  Value pLeft = builder.create<impulse::DynamicSliceOp>(
      loc, positionType, pCkpts, ValueRange{iBody, zeroI64},
      builder.getDenseI64ArrayAttr({1, ctx.positionSize}));
  Value pSumCkptI = builder.create<impulse::DynamicSliceOp>(
      loc, positionType, pSumCkpts, ValueRange{iBody, zeroI64},
      builder.getDenseI64ArrayAttr({1, ctx.positionSize}));

  // Compute subtree momentum sum: pSum - pSumCkpts[i] + pCkpts[i]
  auto pSumMinusCkpt = builder.create<arith::SubFOp>(loc, pSum, pSumCkptI);
  Value subtreePSum = builder.create<arith::AddFOp>(loc, pSumMinusCkpt, pLeft);

  // Check turning
  Value turningAtCkpt = checkTurning(builder, loc, pLeft, p, subtreePSum, ctx);

  Value iNext = builder.create<arith::SubIOp>(loc, iBody, oneI64);
  builder.create<impulse::YieldOp>(loc, ValueRange{iNext, turningAtCkpt});

  builder.setInsertionPointAfter(whileOp);
  return whileOp.getResult(1);
}

std::pair<Value, Value>
impulse::updateCheckpoints(OpBuilder &builder, Location loc, Value leafIdx,
                           Value ckptIdxMax, Value p, Value pSum, Value pCkpts,
                           Value pSumCkpts, const NUTSContext &ctx,
                           bool debugDump) {
  auto i64TensorType = RankedTensorType::get({}, builder.getI64Type());
  auto oneI64 = builder.create<arith::ConstantOp>(
      loc, i64TensorType,
      DenseElementsAttr::get(i64TensorType, builder.getI64IntegerAttr(1)));
  auto zeroI64 = builder.create<arith::ConstantOp>(
      loc, i64TensorType,
      DenseElementsAttr::get(i64TensorType, builder.getI64IntegerAttr(0)));

  Value leafIdxBit0 = builder.create<arith::AndIOp>(loc, leafIdx, oneI64);
  Value isEven = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                       leafIdxBit0, zeroI64);

  auto pCkptsType = cast<RankedTensorType>(pCkpts.getType());

  // Compute updates only on even leafIdx
  SmallVector<Type> ifResultTypes = {pCkptsType, pCkptsType};
  auto ifOp = builder.create<impulse::IfOp>(loc, ifResultTypes, isEven);
  {
    Block *trueBranch = builder.createBlock(&ifOp.getTrueBranch());
    builder.setInsertionPointToStart(trueBranch);

    auto updatedPCkpts = builder.create<impulse::DynamicUpdateSliceOp>(
        loc, pCkptsType, pCkpts, p, ValueRange{ckptIdxMax, zeroI64});
    auto updatedPSumCkpts = builder.create<impulse::DynamicUpdateSliceOp>(
        loc, pCkptsType, pSumCkpts, pSum,
        ValueRange{ckptIdxMax, zeroI64});

    builder.create<impulse::YieldOp>(loc,
                             ValueRange{updatedPCkpts, updatedPSumCkpts});
  }
  {
    Block *falseBranch = builder.createBlock(&ifOp.getFalseBranch());
    builder.setInsertionPointToStart(falseBranch);

    builder.create<impulse::YieldOp>(loc, ValueRange{pCkpts, pSumCkpts});
  }

  builder.setInsertionPointAfter(ifOp);

  Value finalPCkpts = ifOp.getResult(0);
  Value finalPSumCkpts = ifOp.getResult(1);

  return {finalPCkpts, finalPSumCkpts};
}

DualAveragingState impulse::initDualAveraging(OpBuilder &builder, Location loc,
                                              Value stepSize) {
  auto stepSizeType = cast<RankedTensorType>(stepSize.getType());
  auto elemType = stepSizeType.getElementType();
  auto scalarType = RankedTensorType::get({}, elemType);
  auto i64TensorType = RankedTensorType::get({}, builder.getI64Type());

  // prox_center = log(10) + log(step_size)
  auto log10Const = builder.create<arith::ConstantOp>(
      loc, scalarType,
      DenseElementsAttr::get(scalarType,
                             builder.getFloatAttr(elemType, std::log(10.0))));
  Value logStepSize = builder.create<math::LogOp>(loc, stepSize);
  Value proxCenter =
      builder.create<arith::AddFOp>(loc, log10Const, logStepSize);

  auto zeroConst = builder.create<arith::ConstantOp>(
      loc, scalarType,
      DenseElementsAttr::get(scalarType, builder.getFloatAttr(elemType, 0.0)));
  auto zeroI64 = builder.create<arith::ConstantOp>(
      loc, i64TensorType,
      DenseElementsAttr::get(i64TensorType, builder.getI64IntegerAttr(0)));

  return {
      .log_step_size = zeroConst,
      .log_step_size_avg = zeroConst,
      .gradient_avg = zeroConst,
      .step_count = zeroI64,
      .prox_center = proxCenter,
  };
}

DualAveragingState
impulse::updateDualAveraging(OpBuilder &builder, Location loc,
                             const DualAveragingState &state, Value acceptProb,
                             const DualAveragingConfig &config) {
  // Dual Averaging update:
  //   g = target_accept_prob - accept_prob
  //   g_avg = (1 - 1/(t+t0)) * g_avg + g/(t+t0)
  //   log_step_size = prox_center - sqrt(t)/gamma * g_avg
  //   log_step_size_avg = (1 - t^(-kappa)) * log_step_size_avg +
  //                        t^(-kappa) * log_step_size
  auto acceptProbType = cast<RankedTensorType>(acceptProb.getType());
  auto elemType = acceptProbType.getElementType();
  auto scalarType = RankedTensorType::get({}, elemType);
  auto i64TensorType = RankedTensorType::get({}, builder.getI64Type());

  // t = t + 1
  auto oneI64 = builder.create<arith::ConstantOp>(
      loc, i64TensorType,
      DenseElementsAttr::get(i64TensorType, builder.getI64IntegerAttr(1)));
  Value tNew = builder.create<arith::AddIOp>(loc, state.step_count, oneI64);
  Value tFloat = builder.create<arith::SIToFPOp>(loc, scalarType, tNew);

  // t0
  auto t0Const = builder.create<arith::ConstantOp>(
      loc, scalarType,
      DenseElementsAttr::get(scalarType,
                             builder.getFloatAttr(elemType, config.t0)));

  // g = target_accept_prob - accept_prob
  auto targetConst = builder.create<arith::ConstantOp>(
      loc, scalarType,
      DenseElementsAttr::get(
          scalarType,
          builder.getFloatAttr(elemType, config.target_accept_prob)));
  Value g = builder.create<arith::SubFOp>(loc, targetConst, acceptProb);

  // t_plus_t0 = t + t0
  Value tPlusT0 = builder.create<arith::AddFOp>(loc, tFloat, t0Const);

  // weight = 1 / (t + t0)
  auto oneConst = builder.create<arith::ConstantOp>(
      loc, scalarType,
      DenseElementsAttr::get(scalarType, builder.getFloatAttr(elemType, 1.0)));
  Value weight = builder.create<arith::DivFOp>(loc, oneConst, tPlusT0);

  // decay = 1 - weight = 1 - 1/(t + t0)
  Value decay = builder.create<arith::SubFOp>(loc, oneConst, weight);

  // g_avg = decay * g_avg + weight * g
  // g_avg = (1 - 1/(t+t0)) * g_avg + g/(t+t0)
  Value gAvgDecayed =
      builder.create<arith::MulFOp>(loc, decay, state.gradient_avg);
  Value gWeighted = builder.create<arith::MulFOp>(loc, weight, g);
  Value gAvgNew = builder.create<arith::AddFOp>(loc, gAvgDecayed, gWeighted);

  // x_t = prox_center - sqrt(t) / gamma * g_avg
  Value sqrtT = builder.create<math::SqrtOp>(loc, tFloat);
  auto gammaConst = builder.create<arith::ConstantOp>(
      loc, scalarType,
      DenseElementsAttr::get(scalarType,
                             builder.getFloatAttr(elemType, config.gamma)));
  Value sqrtTOverGamma = builder.create<arith::DivFOp>(loc, sqrtT, gammaConst);
  Value adjustment =
      builder.create<arith::MulFOp>(loc, sqrtTOverGamma, gAvgNew);
  Value logStepSizeNew =
      builder.create<arith::SubFOp>(loc, state.prox_center, adjustment);

  // weight_t = t^(-kappa)
  auto negKappaConst = builder.create<arith::ConstantOp>(
      loc, scalarType,
      DenseElementsAttr::get(scalarType,
                             builder.getFloatAttr(elemType, -config.kappa)));
  Value weightT = builder.create<math::PowFOp>(loc, tFloat, negKappaConst);

  // x_avg = (1 - weight_t) * x_avg + weight_t * x_t
  Value oneMinusWeightT =
      builder.create<arith::SubFOp>(loc, oneConst, weightT);
  Value avgDecayed = builder.create<arith::MulFOp>(loc, oneMinusWeightT,
                                           state.log_step_size_avg);
  Value newContribution =
      builder.create<arith::MulFOp>(loc, weightT, logStepSizeNew);
  Value logStepSizeAvgNew =
      builder.create<arith::AddFOp>(loc, avgDecayed, newContribution);

  return {
      .log_step_size = logStepSizeNew,
      .log_step_size_avg = logStepSizeAvgNew,
      .gradient_avg = gAvgNew,
      .step_count = tNew,
      .prox_center = state.prox_center,
  };
}

Value impulse::getStepSizeFromDualAveraging(OpBuilder &builder, Location loc,
                                            const DualAveragingState &state,
                                            bool final) {
  Value logStepSize = final ? state.log_step_size_avg : state.log_step_size;
  return builder.create<math::ExpOp>(loc, logStepSize);
}

WelfordState impulse::initWelford(OpBuilder &builder, Location loc,
                                  int64_t positionSize, bool diagonal) {
  auto elemType = builder.getF64Type();
  auto i64TensorType = RankedTensorType::get({}, builder.getI64Type());
  auto meanType = RankedTensorType::get({positionSize}, elemType);

  Value mean = builder.create<arith::ConstantOp>(
      loc, meanType,
      DenseElementsAttr::get(meanType, builder.getFloatAttr(elemType, 0.0)));

  // Diagonal -> tensor<positionSize>
  // Dense -> tensor<positionSize x positionSize>
  Value m2;
  if (diagonal) {
    m2 = builder.create<arith::ConstantOp>(
        loc, meanType,
        DenseElementsAttr::get(meanType, builder.getFloatAttr(elemType, 0.0)));
  } else {
    auto m2Type = RankedTensorType::get({positionSize, positionSize}, elemType);
    m2 = builder.create<arith::ConstantOp>(
        loc, m2Type,
        DenseElementsAttr::get(m2Type, builder.getFloatAttr(elemType, 0.0)));
  }

  Value n = builder.create<arith::ConstantOp>(
      loc, i64TensorType,
      DenseElementsAttr::get(i64TensorType, builder.getI64IntegerAttr(0)));

  return {mean, m2, n};
}

WelfordState impulse::updateWelford(OpBuilder &builder, Location loc,
                                    const WelfordState &state, Value sample,
                                    const WelfordConfig &config) {
  // Algorithm:
  //   n = n + 1
  //   delta_pre = sample - mean
  //   mean = mean + delta_pre / n
  //   delta_post = sample - mean
  //   (if diagonal) m2 = m2 + delta_pre * delta_post
  //   (if dense)    m2 = m2 + outer(delta_post, delta_pre)

  auto sampleType = cast<RankedTensorType>(sample.getType());
  auto elemType = sampleType.getElementType();
  auto i64TensorType = RankedTensorType::get({}, builder.getI64Type());

  auto oneI64 = builder.create<arith::ConstantOp>(
      loc, i64TensorType,
      DenseElementsAttr::get(i64TensorType, builder.getI64IntegerAttr(1)));
  Value nNew = builder.create<arith::AddIOp>(loc, state.n, oneI64);

  auto scalarType = RankedTensorType::get({}, elemType);
  Value nFloat = builder.create<arith::SIToFPOp>(loc, scalarType, nNew);

  Value nBroadcast = builder.create<BroadcastOp>(loc, sampleType, nFloat,
                                         sampleType.getShape());

  Value deltaPre = builder.create<arith::SubFOp>(loc, sample, state.mean);

  Value deltaPreOverN =
      builder.create<arith::DivFOp>(loc, deltaPre, nBroadcast);
  Value meanNew =
      builder.create<arith::AddFOp>(loc, state.mean, deltaPreOverN);

  Value deltaPost = builder.create<arith::SubFOp>(loc, sample, meanNew);

  Value m2New;
  if (config.diagonal) {
    Value product = builder.create<arith::MulFOp>(loc, deltaPre, deltaPost);
    m2New = builder.create<arith::AddFOp>(loc, state.m2, product);
  } else { // Dense
    auto m2Type = cast<RankedTensorType>(state.m2.getType());
    Value outerProduct = builder.create<impulse::DotOp>(
        loc, m2Type, deltaPost, deltaPre,
        /*lhs_batching_dimensions=*/builder.getDenseI64ArrayAttr({}),
        /*rhs_batching_dimensions=*/builder.getDenseI64ArrayAttr({}),
        /*lhs_contracting_dimensions=*/builder.getDenseI64ArrayAttr({}),
        /*rhs_contracting_dimensions=*/builder.getDenseI64ArrayAttr({}));
    m2New = builder.create<arith::AddFOp>(loc, state.m2, outerProduct);
  }

  return {meanNew, m2New, nNew};
}

Value impulse::finalizeWelford(OpBuilder &builder, Location loc,
                               const WelfordState &state,
                               const WelfordConfig &config) {
  // Compute sample covariance: cov = m2 / (n - 1)
  auto m2Type = cast<RankedTensorType>(state.m2.getType());
  auto elemType = m2Type.getElementType();
  auto scalarType = RankedTensorType::get({}, elemType);
  auto i64TensorType = RankedTensorType::get({}, builder.getI64Type());

  auto oneI64 = builder.create<arith::ConstantOp>(
      loc, i64TensorType,
      DenseElementsAttr::get(i64TensorType, builder.getI64IntegerAttr(1)));
  Value nMinus1 = builder.create<arith::SubIOp>(loc, state.n, oneI64);
  Value nMinus1Float =
      builder.create<arith::SIToFPOp>(loc, scalarType, nMinus1);

  Value nMinus1Bcast = builder.create<BroadcastOp>(loc, m2Type, nMinus1Float,
                                           m2Type.getShape());

  Value cov = builder.create<arith::DivFOp>(loc, state.m2, nMinus1Bcast);

  // (Optional) Regularization (Stan's shrinkage):
  //   scaled_cov = (n / (n + 5)) * cov
  //   shrinkage = 1e-3 * (5 / (n + 5))
  //   (if diagonal) cov = scaled_cov + shrinkage
  //   (if dense)    cov = scaled_cov + shrinkage * I
  if (config.regularize) {
    Value nFloat = builder.create<arith::SIToFPOp>(loc, scalarType, state.n);

    auto fiveConst = builder.create<arith::ConstantOp>(
        loc, scalarType,
        DenseElementsAttr::get(scalarType,
                               builder.getFloatAttr(elemType, 5.0)));
    Value nPlusFive = builder.create<arith::AddFOp>(loc, nFloat, fiveConst);

    Value scale = builder.create<arith::DivFOp>(loc, nFloat, nPlusFive);
    Value scaleBcast =
        builder.create<BroadcastOp>(loc, m2Type, scale, m2Type.getShape());
    Value scaledCov = builder.create<arith::MulFOp>(loc, scaleBcast, cov);

    auto shrinkageBaseConst = builder.create<arith::ConstantOp>(
        loc, scalarType,
        DenseElementsAttr::get(scalarType,
                               builder.getFloatAttr(elemType, 1e-3 * 5.0)));
    Value shrinkage =
        builder.create<arith::DivFOp>(loc, shrinkageBaseConst, nPlusFive);

    if (config.diagonal) {
      Value shrinkageBcast = builder.create<BroadcastOp>(loc, m2Type,
                                                 shrinkage, m2Type.getShape());
      cov = builder.create<arith::AddFOp>(loc, scaledCov, shrinkageBcast);
    } else {
      Value identity = createIdentityMatrix(builder, loc, m2Type);
      Value shrinkageBcast = builder.create<BroadcastOp>(loc, m2Type,
                                                 shrinkage, m2Type.getShape());
      Value shrinkageI =
          builder.create<arith::MulFOp>(loc, shrinkageBcast, identity);
      cov = builder.create<arith::AddFOp>(loc, scaledCov, shrinkageI);
    }
  }

  return cov;
}

SmallVector<AdaptWindow> impulse::buildAdaptationSchedule(int64_t numSteps) {
  // |<-- start buffer -->|<-- middle windows (doubling) -->|<-- end buffer -->|
  // |      (no mass)     |     (collect + adapt mass)      |    (no mass)     |
  // |   step size only   |   step size + mass matrix       |  step size only  |

  SmallVector<AdaptWindow> schedule;

  if (numSteps < 20) {
    schedule.push_back({0, numSteps - 1});
    return schedule;
  }

  // Stan-style window schedule
  int64_t startBufferSize = 75;
  int64_t endBufferSize = 50;
  int64_t initWindowSize = 25;

  if ((startBufferSize + endBufferSize + initWindowSize) > numSteps) {
    startBufferSize = static_cast<int64_t>(0.15 * numSteps);
    endBufferSize = static_cast<int64_t>(0.1 * numSteps);
    initWindowSize = numSteps - startBufferSize - endBufferSize;
  }

  // Start buffer window
  schedule.push_back({0, startBufferSize - 1});

  int64_t endWindowStart = numSteps - endBufferSize;
  int64_t nextWindowSize = initWindowSize;
  int64_t nextWindowStart = startBufferSize;

  // Middle windows
  while (nextWindowStart < endWindowStart) {
    int64_t curWindowStart = nextWindowStart;
    int64_t curWindowSize = nextWindowSize;

    if (3 * curWindowSize <= endWindowStart - curWindowStart) {
      nextWindowSize = 2 * curWindowSize;
    } else {
      curWindowSize = endWindowStart - curWindowStart;
    }

    nextWindowStart = curWindowStart + curWindowSize;
    schedule.push_back({curWindowStart, nextWindowStart - 1});
  }

  // End buffer window
  schedule.push_back({endWindowStart, numSteps - 1});

  return schedule;
}

Value impulse::unconstrainPosition(OpBuilder &builder, Location loc,
                                   Value constrained,
                                   ArrayRef<SupportInfo> supports) {
  bool hasConstraints = false;
  for (const auto &info : supports) {
    if (info.support && info.support.getKind() != impulse::SupportKind::REAL) {
      hasConstraints = true;
      break;
    }
  }
  if (!hasConstraints || supports.empty())
    return constrained;

  auto inputType = cast<RankedTensorType>(constrained.getType());
  auto elemType = inputType.getElementType();
  int64_t positionSize = inputType.getShape()[1];

  auto positionType1D = RankedTensorType::get({positionSize}, elemType);
  Value constrained1D =
      builder.create<impulse::ReshapeOp>(loc, positionType1D, constrained);

  SmallVector<Value> slices;
  for (const auto &info : supports) {
    auto sliceType = RankedTensorType::get({info.size}, elemType);
    auto slice = builder.create<impulse::SliceOp>(
        loc, sliceType, constrained1D,
        builder.getDenseI64ArrayAttr({info.offset}),
        builder.getDenseI64ArrayAttr({info.offset + info.size}),
        builder.getDenseI64ArrayAttr({1}));
    Value unconstrainedSlice;
    if (info.support && info.support.getKind() != impulse::SupportKind::REAL) {
      unconstrainedSlice =
          transforms::unconstrain(builder, loc, slice, info.support);
    } else {
      unconstrainedSlice = slice;
    }

    slices.push_back(unconstrainedSlice);
  }

  Value result1D;
  if (slices.size() == 1) {
    result1D = slices[0];
  } else {
    auto resultType1D = RankedTensorType::get({positionSize}, elemType);
    result1D = builder.create<arith::ConstantOp>(
        loc, resultType1D,
        DenseElementsAttr::get(resultType1D,
                               builder.getFloatAttr(elemType, 0.0)));

    auto i64ScalarType = RankedTensorType::get({}, builder.getI64Type());
    auto elemType1DSlice = RankedTensorType::get({1}, elemType);
    auto elemType0D = RankedTensorType::get({}, elemType);
    int64_t offset = 0;
    for (size_t i = 0; i < slices.size(); ++i) {
      auto sliceType = cast<RankedTensorType>(slices[i].getType());
      int64_t sliceSize = sliceType.getShape()[0];

      for (int64_t j = 0; j < sliceSize; ++j) {
        auto elemIdx = builder.create<arith::ConstantOp>(
            loc, i64ScalarType,
            DenseElementsAttr::get(i64ScalarType,
                                   builder.getI64IntegerAttr(j)));
        auto resultIdx = builder.create<arith::ConstantOp>(
            loc, i64ScalarType,
            DenseElementsAttr::get(i64ScalarType,
                                   builder.getI64IntegerAttr(offset + j)));

        auto elemSliced = builder.create<impulse::DynamicSliceOp>(
            loc, elemType1DSlice, slices[i], ValueRange{elemIdx},
            builder.getDenseI64ArrayAttr({1}));
        auto elem =
            builder.create<impulse::ReshapeOp>(loc, elemType0D, elemSliced);

        auto resultSliced = builder.create<impulse::ReshapeOp>(
            loc, RankedTensorType::get({1}, elemType), elem);
        result1D = builder.create<impulse::DynamicUpdateSliceOp>(
            loc, resultType1D, result1D, resultSliced,
            ValueRange{resultIdx});
      }

      offset += sliceSize;
    }
  }

  auto resultType2D = RankedTensorType::get({1, positionSize}, elemType);
  return builder.create<impulse::ReshapeOp>(loc, resultType2D, result1D);
}

Value impulse::constrainPosition(OpBuilder &builder, Location loc,
                                 Value unconstrained,
                                 ArrayRef<SupportInfo> supports) {
  bool hasConstraints = false;
  for (const auto &info : supports) {
    if (info.support && info.support.getKind() != impulse::SupportKind::REAL) {
      hasConstraints = true;
      break;
    }
  }
  if (!hasConstraints || supports.empty())
    return unconstrained;

  auto inputType = cast<RankedTensorType>(unconstrained.getType());
  auto elemType = inputType.getElementType();
  int64_t positionSize = inputType.getShape()[1];

  auto positionType1D = RankedTensorType::get({positionSize}, elemType);
  Value unconstrained1D =
      builder.create<impulse::ReshapeOp>(loc, positionType1D, unconstrained);

  SmallVector<Value> slices;
  for (const auto &info : supports) {
    auto sliceType = RankedTensorType::get({info.size}, elemType);
    auto slice = builder.create<impulse::SliceOp>(
        loc, sliceType, unconstrained1D,
        builder.getDenseI64ArrayAttr({info.offset}),
        builder.getDenseI64ArrayAttr({info.offset + info.size}),
        builder.getDenseI64ArrayAttr({1}));

    Value constrainedSlice;
    if (info.support && info.support.getKind() != impulse::SupportKind::REAL) {
      constrainedSlice =
          transforms::constrain(builder, loc, slice, info.support);
    } else {
      constrainedSlice = slice;
    }

    slices.push_back(constrainedSlice);
  }

  Value result1D;
  if (slices.size() == 1) {
    result1D = slices[0];
  } else {
    auto resultType1D = RankedTensorType::get({positionSize}, elemType);
    result1D = builder.create<arith::ConstantOp>(
        loc, resultType1D,
        DenseElementsAttr::get(resultType1D,
                               builder.getFloatAttr(elemType, 0.0)));

    auto i64ScalarType = RankedTensorType::get({}, builder.getI64Type());
    auto elemType1DSlice = RankedTensorType::get({1}, elemType);
    auto elemType0D = RankedTensorType::get({}, elemType);
    int64_t offset = 0;
    for (size_t i = 0; i < slices.size(); ++i) {
      auto sliceType = cast<RankedTensorType>(slices[i].getType());
      int64_t sliceSize = sliceType.getShape()[0];

      for (int64_t j = 0; j < sliceSize; ++j) {
        auto elemIdx = builder.create<arith::ConstantOp>(
            loc, i64ScalarType,
            DenseElementsAttr::get(i64ScalarType,
                                   builder.getI64IntegerAttr(j)));
        auto resultIdx = builder.create<arith::ConstantOp>(
            loc, i64ScalarType,
            DenseElementsAttr::get(i64ScalarType,
                                   builder.getI64IntegerAttr(offset + j)));

        auto elemSliced = builder.create<impulse::DynamicSliceOp>(
            loc, elemType1DSlice, slices[i], ValueRange{elemIdx},
            builder.getDenseI64ArrayAttr({1}));
        auto elem =
            builder.create<impulse::ReshapeOp>(loc, elemType0D, elemSliced);

        auto resultSliced =
            builder.create<impulse::ReshapeOp>(loc, elemType1DSlice, elem);
        result1D = builder.create<impulse::DynamicUpdateSliceOp>(
            loc, resultType1D, result1D, resultSliced,
            ValueRange{resultIdx});
      }

      offset += sliceSize;
    }
  }

  auto resultType2D = RankedTensorType::get({1, positionSize}, elemType);
  return builder.create<impulse::ReshapeOp>(loc, resultType2D, result1D);
}

Value impulse::computeTotalJacobianCorrection(OpBuilder &builder, Location loc,
                                              Value unconstrained,
                                              ArrayRef<SupportInfo> supports) {
  auto inputType = cast<RankedTensorType>(unconstrained.getType());
  auto elemType = inputType.getElementType();
  auto scalarType = RankedTensorType::get({}, elemType);

  Value total = builder.create<arith::ConstantOp>(
      loc, scalarType,
      DenseElementsAttr::get(scalarType, builder.getFloatAttr(elemType, 0.0)));

  if (supports.empty())
    return total;

  int64_t positionSize = inputType.getShape()[1];
  auto positionType1D = RankedTensorType::get({positionSize}, elemType);
  Value unconstrained1D =
      builder.create<impulse::ReshapeOp>(loc, positionType1D, unconstrained);

  for (const auto &info : supports) {
    if (!info.support || info.support.getKind() == impulse::SupportKind::REAL)
      continue;

    auto sliceType = RankedTensorType::get({info.size}, elemType);
    auto slice = builder.create<impulse::SliceOp>(
        loc, sliceType, unconstrained1D,
        builder.getDenseI64ArrayAttr({info.offset}),
        builder.getDenseI64ArrayAttr({info.offset + info.size}),
        builder.getDenseI64ArrayAttr({1}));
    auto jacobian =
        transforms::logAbsDetJacobian(builder, loc, slice, info.support);

    total = builder.create<arith::AddFOp>(loc, total, jacobian);
  }

  return total;
}
