//===- LLVMAutoDiffOpInterfaceImpl.cpp - Interface external model  --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the external model implementation of the automatic
// differentiation op interfaces for the upstream LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "Dialect/LLVMExt/LLVMExt.h"
#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "Interfaces/GradientUtils.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace {
#include "Implementations/LLVMDerivatives.inc"

struct InlineAsmActivityInterface
    : public ActivityOpInterface::ExternalModel<InlineAsmActivityInterface,
                                                LLVM::InlineAsmOp> {
  bool isInactive(Operation *op) const {
    auto asmOp = cast<LLVM::InlineAsmOp>(op);
    auto str = asmOp.getAsmString();
    return str.contains("cpuid") || str.contains("exit");
  }
  bool isArgInactive(Operation *op, size_t) const { return isInactive(op); }
};

class PointerTypeInterface
    : public AutoDiffTypeInterface::ExternalModel<PointerTypeInterface,
                                                  LLVM::LLVMPointerType> {
public:
  mlir::Attribute createNullAttr(mlir::Type self) const {
    llvm::errs() << " unsupported: createNullAttribute of pointertype\n";
    return nullptr;
  }

  mlir::Value createNullValue(mlir::Type self, OpBuilder &builder,
                              Location loc) const {
    return builder.create<LLVM::ZeroOp>(loc, self);
  }

  Value createAddOp(Type self, OpBuilder &builder, Location loc, Value a,
                    Value b) const {
    llvm_unreachable("TODO");
  }

  Value createConjOp(Type self, OpBuilder &builder, Location loc,
                     Value a) const {
    llvm_unreachable("TODO");
  }

  Type getShadowType(Type self, unsigned width) const {
    assert(width == 1 && "unsupported width != 1");
    return self;
  }

  bool isMutable(Type self) const { return true; }

  LogicalResult zeroInPlace(Type self, OpBuilder &builder, Location loc,
                            Value val) const {
    // TODO inspect val and memset corresponding size
    return failure();
  }

  bool isZero(Type self, Value val) const { return false; }
  bool isZeroAttr(Type self, Attribute attr) const { return false; }
};

struct GEPOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<GEPOpInterfaceReverse,
                                                       LLVM::GEPOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    return {};
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {
    auto gep = cast<LLVM::GEPOp>(op);
    auto newGep = cast<LLVM::GEPOp>(gutils->getNewFromOriginal(op));
    auto base = gep.getBase();
    if (!gutils->isConstantValue(base)) {
      auto baseShadow = gutils->invertPointerM(base, builder);
      auto shadowGep = cast<LLVM::GEPOp>(builder.clone(*newGep));
      shadowGep.getBaseMutable().assign(baseShadow);
      gutils->setInvertedPointer(gep.getRes(), shadowGep->getResult(0));
    }
  }
};

struct LoadOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<LoadOpInterfaceReverse,
                                                       LLVM::LoadOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto loadOp = cast<LLVM::LoadOp>(op);
    Value addr = loadOp.getAddr();

    if (auto iface = dyn_cast<AutoDiffTypeInterface>(loadOp.getType())) {
      if (!gutils->isConstantValue(loadOp) && !gutils->isConstantValue(addr)) {
        Value gradient = gutils->diffe(loadOp, builder);
        Value addrGradient = gutils->popCache(caches.front(), builder);

        if (!gutils->AtomicAdd) {
          Value loadedGradient = builder.create<LLVM::LoadOp>(loadOp.getLoc(),
                                                      iface, addrGradient);
          Value addedGradient = iface.createAddOp(builder, loadOp.getLoc(),
                                                  loadedGradient, gradient);

          builder.create<LLVM::StoreOp>(loadOp.getLoc(), addedGradient,
                                addrGradient);
        } else {
          builder.create<LLVM::AtomicRMWOp>(loadOp.getLoc(),
                                    LLVM::AtomicBinOp::fadd, addrGradient,
                                    gradient, LLVM::AtomicOrdering::monotonic);
        }
      }
    }

    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    auto loadOp = cast<LLVM::LoadOp>(op);
    auto addr = loadOp.getAddr();
    if (!(isa<AutoDiffTypeInterface>(loadOp.getType()) &&
          (!gutils->isConstantValue(loadOp) && !gutils->isConstantValue(addr))))
      return {};
    OpBuilder cacheBuilder(gutils->getNewFromOriginal(op));
    return {gutils->initAndPushCache(gutils->invertPointerM(addr, cacheBuilder),
                                     cacheBuilder)};
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {
    // auto loadOp = cast<LLVM::LoadOp>(op);
    // Value ptr = loadOp.getAddr();
  }
};

struct StoreOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<StoreOpInterfaceReverse,
                                                       LLVM::StoreOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto storeOp = cast<LLVM::StoreOp>(op);
    Value val = storeOp.getValue();
    Value addr = storeOp.getAddr();

    auto iface = cast<AutoDiffTypeInterface>(val.getType());

    if (!gutils->isConstantValue(addr)) {
      Value addrGradient = gutils->popCache(caches.front(), builder);

      if (!iface.isMutable()) {
        if (!gutils->isConstantValue(val)) {
          Value loadedGradient = builder.create<LLVM::LoadOp>(
              storeOp.getLoc(), val.getType(), addrGradient);
          gutils->addToDiffe(val, loadedGradient, builder);
        }

        auto zero =
            cast<AutoDiffTypeInterface>(gutils->getShadowType(val.getType()))
                .createNullValue(builder, op->getLoc());

        builder.create<LLVM::StoreOp>(storeOp.getLoc(), zero, addrGradient);
      }
    }

    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    auto storeOp = cast<LLVM::StoreOp>(op);
    auto addr = storeOp.getAddr();
    if (gutils->isConstantValue(addr))
      return {};
    OpBuilder cacheBuilder(gutils->getNewFromOriginal(op));
    return {gutils->initAndPushCache(gutils->invertPointerM(addr, cacheBuilder),
                                     cacheBuilder)};
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {}
};

struct ExtractValueOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<
          ExtractValueOpInterfaceReverse, LLVM::ExtractValueOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto evOp = cast<LLVM::ExtractValueOp>(op);
    Value container = evOp.getContainer();

    auto containerIface = dyn_cast<AutoDiffTypeInterface>(container.getType());
    if (!containerIface)
      return failure();

    if (!gutils->isConstantValue(evOp)) {
      Value gradient = gutils->diffe(evOp, builder);
      gutils->zeroDiffe(evOp, builder);
      // Create a zero aggregate matching the container type, then insert the
      // gradient at the extracted position.
      if (!gutils->isConstantValue(container)) {
        Value zero = containerIface.createNullValue(builder, op->getLoc());
        Value grad = builder.create<LLVM::InsertValueOp>(op->getLoc(), zero,
                                                 gradient, evOp.getPosition());
        gutils->addToDiffe(container, grad, builder);
      }
    }

    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    return {};
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {}
};

struct InsertValueOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<
          InsertValueOpInterfaceReverse, LLVM::InsertValueOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto ivOp = cast<LLVM::InsertValueOp>(op);
    Value container = ivOp.getContainer();
    Value value = ivOp.getValue();

    auto resultIface = dyn_cast<AutoDiffTypeInterface>(ivOp.getType());
    if (!resultIface)
      return failure();

    if (!gutils->isConstantValue(ivOp)) {
      Value gradient = gutils->diffe(ivOp, builder);
      gutils->zeroDiffe(ivOp, builder);

      // Propagate gradient to the inserted value: extract from the result
      // gradient at the insertion position.
      auto valIface = dyn_cast<AutoDiffTypeInterface>(value.getType());
      if (valIface && !gutils->isConstantValue(value)) {
        Value valGrad = builder.create<LLVM::ExtractValueOp>(
            op->getLoc(), gradient, ivOp.getPosition());
        gutils->addToDiffe(value, valGrad, builder);
      }

      // Propagate gradient to the container: zero out the inserted position
      // in the result gradient, then add to the container gradient.
      if (!gutils->isConstantValue(container)) {
        Value zeroVal =
            valIface
                ? valIface.createNullValue(builder, op->getLoc())
                : builder.create<LLVM::ZeroOp>(op->getLoc(), value.getType());
        Value containerGrad = builder.create<LLVM::InsertValueOp>(
            op->getLoc(), gradient, zeroVal, ivOp.getPosition());
        gutils->addToDiffe(container, containerGrad, builder);
      }
    }

    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    return {};
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {}
};

std::optional<Value> findPtrSize(Value ptr) {
  if (auto allocOp = ptr.getDefiningOp<llvm_ext::AllocOp>())
    return allocOp.getSize();

  for (auto user : ptr.getUsers()) {
    if (auto psh = dyn_cast<llvm_ext::PtrSizeHintOp>(user)) {
      return psh.getSize();
    }
  }

  return std::nullopt;
}

struct PointerClonableTypeInterface
    : public ClonableTypeInterface::ExternalModel<PointerClonableTypeInterface,
                                                  LLVM::LLVMPointerType> {
  mlir::Value cloneValue(Type self, OpBuilder &builder, Value value) const {
    auto ptrSize = findPtrSize(value);
    if (!ptrSize) {
      llvm::errs() << "cannot find size of ptr: " << value << "\n";
      return nullptr;
    }

    auto clone = builder.create<llvm_ext::AllocOp>(
        value.getLoc(), LLVM::LLVMPointerType::get(value.getContext()),
        *ptrSize);
    builder.create<LLVM::MemcpyOp>(value.getLoc(), clone, value, *ptrSize,
                           /*isVolatile*/ false);

    return clone;
  }

  void freeClonedValue(Type self, OpBuilder &builder, Value value) const {
    builder.create<llvm_ext::FreeOp>(value.getLoc(), value);
  }
};

class StructTypeInterface
    : public AutoDiffTypeInterface::ExternalModel<StructTypeInterface,
                                                  LLVM::LLVMStructType> {
public:
  mlir::Attribute createNullAttr(mlir::Type self) const {
    llvm::errs() << " unsupported: createNullAttribute of LLVMStructType\n";
    return nullptr;
  }

  mlir::Value createNullValue(mlir::Type self, OpBuilder &builder,
                              Location loc) const {
    auto structTy = cast<LLVM::LLVMStructType>(self);
    Value result = builder.create<LLVM::PoisonOp>(loc, structTy);
    for (auto &&[i, elemTy] : llvm::enumerate(structTy.getBody())) {
      auto elemIface = dyn_cast<AutoDiffTypeInterface>(elemTy);
      if (!elemIface) {
        Value zero = builder.create<LLVM::ZeroOp>(loc, elemTy);
        result = builder.create<LLVM::InsertValueOp>(loc, result, zero, i);
        continue;
      }
      Value nullElem = elemIface.createNullValue(builder, loc);
      result = builder.create<LLVM::InsertValueOp>(loc, result, nullElem, i);
    }
    return result;
  }

  Value createAddOp(Type self, OpBuilder &builder, Location loc, Value a,
                    Value b) const {
    auto structTy = cast<LLVM::LLVMStructType>(self);
    Value result = builder.create<LLVM::PoisonOp>(loc, structTy);
    for (auto &&[i, elemTy] : llvm::enumerate(structTy.getBody())) {
      Value aElem = builder.create<LLVM::ExtractValueOp>(loc, a, i);
      Value bElem = builder.create<LLVM::ExtractValueOp>(loc, b, i);
      auto elemIface = dyn_cast<AutoDiffTypeInterface>(elemTy);
      Value sum;
      if (elemIface) {
        sum = elemIface.createAddOp(builder, loc, aElem, bElem);
      } else {
        sum = aElem;
      }
      result = builder.create<LLVM::InsertValueOp>(loc, result, sum, i);
    }
    return result;
  }

  Value createConjOp(Type self, OpBuilder &builder, Location loc,
                     Value a) const {
    llvm_unreachable("TODO");
  }

  Type getShadowType(Type self, unsigned width) const {
    assert(width == 1 && "unsupported width != 1");
    return self;
  }

  bool isMutable(Type self) const { return false; }

  LogicalResult zeroInPlace(Type self, OpBuilder &builder, Location loc,
                            Value val) const {
    return failure();
  }

  bool isZero(Type self, Value val) const { return false; }
  bool isZeroAttr(Type self, Attribute attr) const { return false; }
};

static Value packIntoStruct(ValueRange values, OpBuilder &builder,
                            Location loc) {
  SmallVector<Type> resultTypes =
      llvm::map_to_vector(values, [](Value v) { return v.getType(); });
  auto structType =
      LLVM::LLVMStructType::getLiteral(builder.getContext(), resultTypes);
  Value result = builder.create<LLVM::PoisonOp>(loc, structType);
  for (auto &&[i, v] : llvm::enumerate(values))
    result = builder.create<LLVM::InsertValueOp>(loc, result, v, i);

  return result;
}

class AutoDiffLLVMFuncOpFunctionInterface
    : public AutoDiffFunctionInterface::ExternalModel<
          AutoDiffLLVMFuncOpFunctionInterface, LLVM::LLVMFuncOp> {
public:
  void transformResultTypes(Operation *self,
                            SmallVectorImpl<Type> &resultTypes) const {
    auto fn = cast<mlir::FunctionOpInterface>(self);
    auto FTy = fn.getFunctionType();
    if (resultTypes.empty()) {
      // llvm.func ops that return no results need to explicitly return
      // LLVMVoidType
      resultTypes.push_back(LLVM::LLVMVoidType::get(FTy.getContext()));
    } else if (resultTypes.size() > 1) {
      auto structType =
          LLVM::LLVMStructType::getLiteral(FTy.getContext(), resultTypes);
      resultTypes.clear();
      resultTypes.push_back(structType);
    }
  }

  Operation *createCall(Operation *self, OpBuilder &builder, Location loc,
                        ValueRange args) const {
    return builder.create<LLVM::CallOp>(loc, cast<LLVM::LLVMFuncOp>(self),
                                args);
  }

  Operation *createReturn(Operation *self, OpBuilder &builder, Location loc,
                          ValueRange retargs) const {
    if (retargs.size() > 1) {
      Value packedReturns = packIntoStruct(retargs, builder, loc);
      return builder.create<LLVM::ReturnOp>(loc, packedReturns);
    }

    return builder.create<LLVM::ReturnOp>(loc, retargs);
  }
};

} // namespace

void mlir::enzyme::registerLLVMDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, LLVM::LLVMDialect *) {
    registerInterfaces(context);
    LLVM::LLVMPointerType::attachInterface<PointerTypeInterface>(*context);
    LLVM::LLVMPointerType::attachInterface<PointerClonableTypeInterface>(
        *context);
    LLVM::LLVMStructType::attachInterface<StructTypeInterface>(*context);
    LLVM::LoadOp::attachInterface<LoadOpInterfaceReverse>(*context);
    LLVM::StoreOp::attachInterface<StoreOpInterfaceReverse>(*context);
    LLVM::GEPOp::attachInterface<GEPOpInterfaceReverse>(*context);
    LLVM::ExtractValueOp::attachInterface<ExtractValueOpInterfaceReverse>(
        *context);
    LLVM::InsertValueOp::attachInterface<InsertValueOpInterfaceReverse>(
        *context);
    LLVM::UnreachableOp::template attachInterface<
        detail::NoopRevAutoDiffInterface<LLVM::UnreachableOp>>(*context);
    LLVM::LLVMFuncOp::attachInterface<AutoDiffLLVMFuncOpFunctionInterface>(
        *context);
  });
}
