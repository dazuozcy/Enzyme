# `--enzyme-wrap` Pass 详解

## 一、概述

`--enzyme-wrap` 是 Enzyme MLIR 中用于**对整个函数进行自动微分**的 pass。它直接克隆目标函数，对克隆体执行微分变换，生成一个新的导数函数。

与 `--enzyme` pass 的区别：

- **`--enzyme-wrap`**：函数级变换，直接在 IR 中生成导数函数，不产生 `enzyme.autodiff` op
- **`--enzyme`**：调用点级变换，将 IR 中已有的 `enzyme.autodiff` / `enzyme.fwddiff` op 替换为对导数函数的 `func.call`

两者共享同一个底层微分引擎（`MEnzymeLogic::CreateForwardDiff` / `CreateReverseDiff`）。

---

## 二、CLI 参数

定义在 `Enzyme/MLIR/Passes/Passes.td:82-133`：

```bash
--enzyme-wrap="infn=<name> outfn=<name> argTys=<activities> retTys=<activities> mode=<mode>"
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `infn` | string | `""` | 要微分的函数名。为空时取 module 中第一个函数 |
| `outfn` | string | `""` | 输出函数名。为空时**删除原函数**，导数函数继承 `infn` 的名字 |
| `argTys` | string | `""` | 逗号分隔的参数活性标注 |
| `retTys` | string | `""` | 逗号分隔的返回值活性标注 |
| `mode` | enum | `ForwardMode` | 微分模式 |

---

## 三、微分模式（DerivativeMode）

定义在 `Enzyme/Utils.h:389-396`：

```cpp
enum class DerivativeMode {
  ForwardMode = 0,            // 前向模式：同时计算原值和切线
  ReverseModePrimal = 1,      // 反向模式的前向阶段（仅缓存中间值）
  ReverseModeGradient = 2,    // 反向模式的反向阶段（仅计算梯度）
  ReverseModeCombined = 3,    // 反向模式完整流程（前向+反向合一）
  ForwardModeSplit = 4,       // 前向模式拆分变体
  ForwardModeError = 5,       // 前向误差模式
};
```

---

## 四、活性标注（Activity）

### 4.1 内部枚举 `DIFFE_TYPE`（`Utils.h:375-386`）

```cpp
enum class DIFFE_TYPE {
  OUT_DIFF = 0,    // enzyme_active: 标量活跃参数，梯度通过返回值传出
  DUP_ARG = 1,     // enzyme_dup:  复制参数，原值+影子（切线/余切线）成对传入
  CONSTANT = 2,    // enzyme_const: 常量参数，不参与微分
  DUP_NONEED = 3,  // enzyme_dupnoneed: 同 DUP_ARG，但不需要原值输出
};
```

### 4.2 TableGen 枚举 `Activity`（`EnzymeEnums.td`）

用于 `enzyme.autodiff` / `enzyme.fwddiff` op 的属性，比 `DIFFE_TYPE` 多两个：

| 标注 | 值 | 含义 |
|------|----|------|
| `enzyme_active` | 0 | 活跃：参与微分，梯度通过返回值/参数传递 |
| `enzyme_dup` | 1 | 复制：原值 + 影子成对传入 |
| `enzyme_const` | 2 | 常量：不参与微分 |
| `enzyme_dupnoneed` | 3 | 复制但不需要原值输出 |
| `enzyme_activenoneed` | 4 | 活跃但不需要原值输出（仅 `--enzyme` 可用） |
| `enzyme_constnoneed` | 5 | 常量且不需要输出（仅 `--enzyme` 可用） |

> **注意**：`--enzyme-wrap` 的 `parseActivityString`（`EnzymeWrapPass.cpp:38-59`）只识别前 4 种。

### 4.3 字符串解析逻辑

```cpp
// EnzymeWrapPass.cpp:38-59
std::vector<DIFFE_TYPE> parseActivityString(StringRef inp) {
  if (inp.size() == 0) return {};
  std::vector<DIFFE_TYPE> ArgActivity;
  SmallVector<StringRef, 1> split;
  StringRef(inp.data(), inp.size()).split(split, ',');
  for (auto &str : split) {
    if (str == "enzyme_dup")
      ArgActivity.push_back(DIFFE_TYPE::DUP_ARG);
    else if (str == "enzyme_const")
      ArgActivity.push_back(DIFFE_TYPE::CONSTANT);
    else if (str == "enzyme_dupnoneed")
      ArgActivity.push_back(DIFFE_TYPE::DUP_NONEED);
    else if (str == "enzyme_active")
      ArgActivity.push_back(DIFFE_TYPE::OUT_DIFF);
    else {
      llvm::errs() << "unknown activity to parse, found: '" << str << "'\n";
      assert(0 && " unknown constant");
    }
  }
  return ArgActivity;
}
```

---

## 五、执行流程（`EnzymeWrapPass.cpp:82-166`）

```
runOnOperation()
│
├── 1. 查找目标函数 (infn)
│   ├── infn != "" → symbolTable.lookupSymbolIn(infn)
│   └── infn == "" → 遍历 module 取第一个 FunctionOpInterface
│
├── 2. 解析 argTys → std::vector<DIFFE_TYPE> ArgActivity
│   └── 验证: ArgActivity.size() == fn.getNumArguments()
│
├── 3. 解析 retTys → std::vector<DIFFE_TYPE> RetActivity
│   └── 验证: RetActivity.size() == fn.getNumResults()
│
├── 4. 计算 returnPrimal / returnShadow
│   ├── returnPrimal[i] = (RetActivity[i] == DUP_ARG)
│   └── returnShadow[i] = false
│
├── 5. 计算 volatile_args
│   └── volatile_args[i] = !(mode == ReverseModeCombined)
│
├── 6. 根据 mode 分支
│   ├── ForwardMode → Logic.CreateForwardDiff(...)
│   └── Reverse*    → Logic.CreateReverseDiff(...)
│
└── 7. 命名输出函数
    ├── outfn == "" → 删除原函数，导数函数继承 infn 名，设为 Public
    └── outfn != "" → 保留原函数，导数函数命名为 outfn，设为 Private
```

---

## 六、导数函数签名的构造规则

核心逻辑在 `CloneFunction.cpp:17-73` 的 `getFunctionTypeForClone` 中：

```cpp
template <typename T>
mlir::FunctionType
getFunctionTypeForClone(T FTy, DerivativeMode mode, unsigned width,
                        mlir::Type additionalArg,
                        const std::vector<bool> &returnPrimals,
                        const std::vector<bool> &returnShadows,
                        llvm::ArrayRef<DIFFE_TYPE> ReturnActivity,
                        llvm::ArrayRef<DIFFE_TYPE> ArgActivity) {
  // ...
  SmallVector<mlir::Type, 4> RetTypes;
  SmallVector<mlir::Type, 4> ArgTypes;

  // 构造返回值列表
  for (auto &&[Ty, returnPrimal, returnShadow, activity] :
       llvm::zip(origResultTypes, returnPrimals, returnShadows, ReturnActivity)) {
    if (returnPrimal)  RetTypes.push_back(Ty);
    if (returnShadow)  RetTypes.push_back(getShadowType(Ty, width));
  }

  // 构造参数列表
  for (auto &&[ITy, act] : llvm::zip(origInputTypes, ArgActivity)) {
    ArgTypes.push_back(ITy);
    if (act == DIFFE_TYPE::DUP_ARG || act == DIFFE_TYPE::DUP_NONEED)
      ArgTypes.push_back(getShadowType(ITy, width));
    else if (act == DIFFE_TYPE::OUT_DIFF)
      RetTypes.push_back(getShadowType(ITy, width));  // 梯度走返回值
  }

  // 返回值中 OUT_DIFF 对应的种子梯度作为参数传入
  for (auto &&[Ty, activity] : llvm::zip(origResultTypes, ReturnActivity)) {
    if (activity == DIFFE_TYPE::OUT_DIFF)
      ArgTypes.push_back(getShadowType(Ty, width));
  }

  if (additionalArg) ArgTypes.push_back(additionalArg);
  return builder.getFunctionType(ArgTypes, RetTypes);
}
```

### 6.1 参数列表构造

对于原函数的每个参数 `(type_i, activity_i)`：

| activity | 添加到新参数列表 |
|----------|------------------|
| `enzyme_const` | `type_i`（仅原值） |
| `enzyme_active` | `type_i`（仅原值，梯度走返回值） |
| `enzyme_dup` | `type_i, shadow(type_i)`（原值 + 影子） |
| `enzyme_dupnoneed` | `type_i, shadow(type_i)`（同 dup） |

然后，对于原函数的每个**返回值** `(retType_j, retActivity_j)`：

- 若 `retActivity_j == OUT_DIFF`：追加 `shadow(retType_j)` 到参数列表（这是**种子梯度/余切线**）

### 6.2 返回值列表构造

对于原函数的每个返回值 `(retType_i, returnPrimal_i, returnShadow_i)`：

- 若 `returnPrimal_i` 为 true → 追加 `retType_i`
- 若 `returnShadow_i` 为 true → 追加 `shadow(retType_i)`

然后，对于原函数的每个**参数** `(argType_k, argActivity_k)`：

- 若 `argActivity_k == OUT_DIFF`：追加 `shadow(argType_k)` 到返回值列表（这是该参数的**梯度**）

### 6.3 具体示例

**原函数**：

```mlir
func.func @f(
  %X: memref<4x2xf32>,
  %W: memref<2x1xf32>,
  %b: memref<1xf32>,
  %y: memref<4x1xf32>
) -> f32
```

**微分调用**：

```bash
--enzyme-wrap="infn=f outfn=df argTys=enzyme_const,enzyme_dup,enzyme_dup,enzyme_const retTys=enzyme_active mode=ReverseModeCombined"
```

**签名推导**：

参数列表：

| 原参数 | activity | 新参数 |
|--------|----------|--------|
| `X: memref<4x2xf32>` | `enzyme_const` | `memref<4x2xf32>` |
| `W: memref<2x1xf32>` | `enzyme_dup` | `memref<2x1xf32>`, `memref<2x1xf32>` (影子 dW) |
| `b: memref<1xf32>` | `enzyme_dup` | `memref<1xf32>`, `memref<1xf32>` (影子 db) |
| `y: memref<4x1xf32>` | `enzyme_const` | `memref<4x1xf32>` |
| 返回值 `f32` | `enzyme_active` | `f32` (种子梯度) |

返回值列表：

- `returnPrimal = false`（`enzyme_active` 不是 `DUP_ARG`），所以**不返回原值**
- 无 `OUT_DIFF` 参数，所以**不通过返回值传梯度**（梯度写入影子 memref）

**最终签名**：

```mlir
func.func private @df(
  %X: memref<4x2xf32>,
  %W: memref<2x1xf32>, %dW: memref<2x1xf32>,
  %b: memref<1xf32>,   %db: memref<1xf32>,
  %y: memref<4x1xf32>,
  %seed: f32
) -> ()
```

---

## 七、`outfn` 参数的行为

```cpp
// EnzymeWrapPass.cpp:151-163
if (outfn == "") {
  fn->erase();                                          // 删除原函数
  SymbolTable::setSymbolVisibility(newFunc, Public);    // 导数函数设为公开
  SymbolTable::setSymbolName(newFunc, (std::string)infn); // 继承原名
} else {
  SymbolTable::setSymbolName(newFunc, (std::string)outfn); // 导数函数用新名
  // 原函数保留，导数函数默认为 Private
}
```

| `outfn` 值 | 原函数 | 导数函数名 | 导数函数可见性 |
|------------|--------|-----------|----------------|
| `""` (空) | **删除** | 继承 `infn` | Public |
| 非空 | **保留** | `outfn` 指定 | Private |

---

## 八、与 `--enzyme` pass 的关系

| 维度 | `--enzyme-wrap` | `--enzyme` |
|------|-----------------|------------|
| 操作粒度 | 函数级 | 调用点级 |
| 输入 | 普通 `func.func` | 包含 `enzyme.autodiff`/`enzyme.fwddiff` op 的 IR |
| 输出 | 新的导数函数 | 将 autodiff op 替换为 `func.call` |
| 活性信息来源 | CLI 字符串 `argTys`/`retTys` | op 上的 `activity`/`ret_activity` 属性 |
| 是否生成中间 op | 否，直接生成导数函数 | 否，消费已有 op |
| 典型用法 | 从命令行一键微分 | 前端/手写 IR 中嵌入 autodiff op 后降级 |

`--enzyme` pass 的核心逻辑（`EnzymeMLIRPass.cpp:356-405`）：

1. 遍历所有函数，找到 `enzyme.fwddiff` 和 `enzyme.autodiff` op
2. 递归处理被引用的函数（先微分内层函数）
3. 调用同一个 `CreateForwardDiff`/`CreateReverseDiff` 生成导数函数
4. 将 autodiff op 替换为 `func.call @导数函数(...)`

---

## 九、底层微分引擎调用链

```
--enzyme-wrap
  └── runOnOperation()
        ├── [ForwardMode] → MEnzymeLogic::CreateForwardDiff()
        │     └── MDiffeGradientUtils::CreateFromClone()
        │           ├── CloneFunctionWithReturns()      // 克隆函数并构造新签名
        │           │     └── getFunctionTypeForClone()  // 签名构造
        │           ├── forceAugmentedReturns()           // 处理增广返回
        │           └── for each op: visitChild()         // 逐 op 前向微分
        │
        └── [ReverseMode*] → MEnzymeLogic::CreateReverseDiff()
              └── MGradientUtilsReverse::CreateFromClone()
                    ├── CloneFunctionWithReturns()       // 克隆函数并构造新签名
                    ├── forceAugmentedReturns()           // 处理增广返回
                    └── differentiate()                   // 反向遍历，计算余切线
```

---

## 十、关键源文件索引

| 文件 | 行号 | 内容 |
|------|------|------|
| `Enzyme/MLIR/Passes/Passes.td` | 82-133 | `--enzyme-wrap` pass 的 TableGen 定义 |
| `Enzyme/MLIR/Passes/EnzymeWrapPass.cpp` | 全文 (166 行) | pass 实现，含 `parseActivityString` |
| `Enzyme/MLIR/Dialect/EnzymeEnums.td` | 全文 (36 行) | `Activity` 枚举定义 (6 种状态) |
| `Enzyme/MLIR/Dialect/EnzymeOps.td` | 99-161 | `ForwardDiffOp` 和 `AutoDiffOp` 定义 |
| `Enzyme/MLIR/Dialect/Ops.h` | 41-95 | `filterGradInputs` 辅助函数 (输入布局约定) |
| `Enzyme/Utils.h` | 375-398 | `DIFFE_TYPE` 和 `DerivativeMode` 枚举 |
| `Enzyme/MLIR/Interfaces/CloneFunction.cpp` | 17-73, 203-415 | 签名构造 + 函数克隆 |
| `Enzyme/MLIR/Interfaces/EnzymeLogic.cpp` | 78-218 | `CreateForwardDiff` 实现 |
| `Enzyme/MLIR/Interfaces/EnzymeLogicReverse.cpp` | 192-289 | `CreateReverseDiff` 实现 |
| `Enzyme/MLIR/Interfaces/GradientUtils.h` | 150-195 | `MDiffeGradientUtils::CreateFromClone` (前向) |
| `Enzyme/MLIR/Interfaces/GradientUtilsReverse.cpp` | 138-182 | `MGradientUtilsReverse::CreateFromClone` (反向) |
| `Enzyme/MLIR/Passes/EnzymeMLIRPass.cpp` | 356-415 | `--enzyme` pass 实现 |
