//===- EnzymeAcitivtyOpt.cpp - Optimize activity for differentiation -------- //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/Ops.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "Interfaces/Utils.h"
#include "PassDetails.h"
#include "Passes/Passes.h"
#include "Passes/Utils.h"

#define DEBUG_TYPE "enzyme-activity-opt"
#define ENZYME_DBGS llvm::dbgs() << "[" << DEBUG_TYPE << "]"

using namespace mlir;
using namespace mlir::enzyme;
using namespace enzyme;

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_ENZYMEACTIVITYOPT
#include "Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

namespace {
struct EnzymeActivityOptPass
    : public enzyme::impl::EnzymeActivityOptBase<EnzymeActivityOptPass> {
    using EnzymeActivityOptBase::EnzymeActivityOptBase;
  void runOnOperation() override { return; }
};
} // namespace
