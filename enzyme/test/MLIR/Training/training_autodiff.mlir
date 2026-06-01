// =============================================================================
// End-to-end Training Pipeline with enzyme.autodiff
// =============================================================================
//
// Pipeline:
//   1. forward_loss: tensor 语义的前向+损失函数 (见 autodiff/forward_loss_tensor.mlir)
//      经 one-shot-bufferize 后等价于下方的 memref 版本
//   2. enzyme.autodiff: 显式的自动微分 op，引用 forward_loss
//   3. --enzyme pass: 展开 enzyme.autodiff 为梯度计算函数
//   4. 通用 lowering passes: scf->cf, memref->LLVM, arith->LLVM, func->LLVM
//   5. mlir-cpu-runner JIT 执行
//
// 训练目标: y = X @ W + b
//   X = [[1,0],[0,1],[1,1],[1,-1]]
//   y_true = [[2],[3],[5],[-1]]
//   最优解: W = [[2],[3]], b = [0]
// =============================================================================

module {
  // ===========================================================================
  // Forward pass + MSE Loss (memref 语义, 等价于 tensor 版 bufferize 后的结果)
  // ===========================================================================
  func.func @forward_loss(
    %X: memref<4x2xf32>,
    %W: memref<2x1xf32>,
    %b: memref<1xf32>,
    %y_true: memref<4x1xf32>
  ) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %cst_zero = arith.constant 0.0 : f32
    %cst_4 = arith.constant 4.0 : f32

    %sum_sq = scf.for %i = %c0 to %c4 step %c1 iter_args(%s1 = %cst_zero) -> f32 {
      %s2 = scf.for %j = %c0 to %c1 step %c1 iter_args(%s1_ = %s1) -> f32 {
        %bval = memref.load %b[%j] : memref<1xf32>
        %dot = scf.for %k = %c0 to %c2 step %c1 iter_args(%acc = %cst_zero) -> f32 {
          %xv = memref.load %X[%i, %k] : memref<4x2xf32>
          %wv = memref.load %W[%k, %j] : memref<2x1xf32>
          %prod = arith.mulf %xv, %wv : f32
          %nacc = arith.addf %acc, %prod : f32
          scf.yield %nacc : f32
        }
        %pred = arith.addf %dot, %bval : f32
        %tv = memref.load %y_true[%i, %j] : memref<4x1xf32>
        %diff = arith.subf %pred, %tv : f32
        %sq = arith.mulf %diff, %diff : f32
        %ns = arith.addf %s1_, %sq : f32
        scf.yield %ns : f32
      }
      scf.yield %s2 : f32
    }

    %mse = arith.divf %sum_sq, %cst_4 : f32
    return %mse : f32
  }

  // ===========================================================================
  // 梯度计算: 通过 enzyme.autodiff op 调用 forward_loss 的反向模式自动微分
  //
  // activity 标注 (对应原函数 forward_loss 的 4 个参数):
  //   X:       enzyme_const  (输入数据, 不求梯度)
  //   W:       enzyme_dup    (权重, 求梯度, 需要 shadow memref dW)
  //   b:       enzyme_dup    (偏置, 求梯度, 需要 shadow memref db)
  //   y_true:  enzyme_const  (标签, 不求梯度)
  //
  // ret_activity:
  //   loss:    enzyme_active (标量损失, 需要种子梯度)
  //
  // 输入布局: (X, W, dW, b, db, y_true, seed)
  //   - enzyme_dup 参数: primal + shadow 成对出现
  //   - enzyme_active 返回值: 种子梯度追加在末尾
  //
  // 输出: (dW_grad, db_grad) — 对应 enzyme_dup 参数的梯度
  // ===========================================================================
  func.func @compute_gradients(
    %X: memref<4x2xf32>,
    %W: memref<2x1xf32>,
    %b: memref<1xf32>,
    %y_true: memref<4x1xf32>,
    %dW: memref<2x1xf32>,
    %db: memref<1xf32>
  ) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst_zero = arith.constant 0.0 : f32
    %seed = arith.constant 1.0 : f32

    // 清零 dW 和 db (enzyme.autodiff 在 shadow 上累加梯度)
    scf.for %k = %c0 to %c2 step %c1 {
      scf.for %j = %c0 to %c1 step %c1 {
        memref.store %cst_zero, %dW[%k, %j] : memref<2x1xf32>
      }
    }
    scf.for %j = %c0 to %c1 step %c1 {
      memref.store %cst_zero, %db[%j] : memref<1xf32>
    }

    enzyme.autodiff @forward_loss(%X, %W, %dW, %b, %db, %y_true, %seed) {
      activity = [#enzyme<activity enzyme_const>, #enzyme<activity enzyme_dup>, #enzyme<activity enzyme_dup>, #enzyme<activity enzyme_const>],
      ret_activity = [#enzyme<activity enzyme_active>]
    } : (memref<4x2xf32>, memref<2x1xf32>, memref<2x1xf32>, memref<1xf32>, memref<1xf32>, memref<4x1xf32>, f32) -> ()
    return
  }

  // ===========================================================================
  // SGD 更新: W -= lr * dW, b -= lr * db
  // ===========================================================================
  func.func @sgd_update(
    %W: memref<2x1xf32>,
    %b: memref<1xf32>,
    %dW: memref<2x1xf32>,
    %db: memref<1xf32>,
    %lr: f32
  ) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    scf.for %k = %c0 to %c2 step %c1 {
      scf.for %j = %c0 to %c1 step %c1 {
        %w = memref.load %W[%k, %j] : memref<2x1xf32>
        %dw = memref.load %dW[%k, %j] : memref<2x1xf32>
        %scaled = arith.mulf %lr, %dw : f32
        %w_new = arith.subf %w, %scaled : f32
        memref.store %w_new, %W[%k, %j] : memref<2x1xf32>
      }
    }
    scf.for %j = %c0 to %c1 step %c1 {
      %bi = memref.load %b[%j] : memref<1xf32>
      %dbi = memref.load %db[%j] : memref<1xf32>
      %scaled = arith.mulf %lr, %dbi : f32
      %b_new = arith.subf %bi, %scaled : f32
      memref.store %b_new, %b[%j] : memref<1xf32>
    }
    return
  }

  // ===========================================================================
  // 外部打印函数 (C runtime)
  // ===========================================================================
  func.func private @print_f32(f32)
  func.func private @print_newline()

  // ===========================================================================
  // Main: 初始化数据, 训练循环, 打印结果
  // ===========================================================================
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %num_epochs = arith.constant 500 : index
    %c50 = arith.constant 50 : index
    %cst_zero = arith.constant 0.0 : f32
    %lr = arith.constant 0.05 : f32

    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %i3 = arith.constant 3 : index

    %X = memref.alloc() : memref<4x2xf32>
    %W = memref.alloc() : memref<2x1xf32>
    %b = memref.alloc() : memref<1xf32>
    %y_true = memref.alloc() : memref<4x1xf32>
    %dW = memref.alloc() : memref<2x1xf32>
    %db = memref.alloc() : memref<1xf32>

    // X = [[1,0],[0,1],[1,1],[1,-1]]
    %x00 = arith.constant 1.0 : f32
    %x01 = arith.constant 0.0 : f32
    %x11 = arith.constant 1.0 : f32
    %x31 = arith.constant -1.0 : f32
    memref.store %x00, %X[%i0, %i0] : memref<4x2xf32>
    memref.store %x01, %X[%i0, %i1] : memref<4x2xf32>
    memref.store %x01, %X[%i1, %i0] : memref<4x2xf32>
    memref.store %x11, %X[%i1, %i1] : memref<4x2xf32>
    memref.store %x00, %X[%i2, %i0] : memref<4x2xf32>
    memref.store %x11, %X[%i2, %i1] : memref<4x2xf32>
    memref.store %x00, %X[%i3, %i0] : memref<4x2xf32>
    memref.store %x31, %X[%i3, %i1] : memref<4x2xf32>

    // y_true = [[2],[3],[5],[-1]]
    %y0 = arith.constant 2.0 : f32
    %y1 = arith.constant 3.0 : f32
    %y2 = arith.constant 5.0 : f32
    %y3 = arith.constant -1.0 : f32
    memref.store %y0, %y_true[%i0, %i0] : memref<4x1xf32>
    memref.store %y1, %y_true[%i1, %i0] : memref<4x1xf32>
    memref.store %y2, %y_true[%i2, %i0] : memref<4x1xf32>
    memref.store %y3, %y_true[%i3, %i0] : memref<4x1xf32>

    // W = [[0.5],[0.5]], b = [0.0]
    %w_init = arith.constant 0.5 : f32
    memref.store %w_init, %W[%i0, %i0] : memref<2x1xf32>
    memref.store %w_init, %W[%i1, %i0] : memref<2x1xf32>
    memref.store %cst_zero, %b[%i0] : memref<1xf32>

    // 训练循环
    scf.for %epoch = %c0 to %num_epochs step %c1 {
      %loss = func.call @forward_loss(%X, %W, %b, %y_true) : (memref<4x2xf32>, memref<2x1xf32>, memref<1xf32>, memref<4x1xf32>) -> f32

      %rem = arith.remsi %epoch, %c50 : index
      %is_print = arith.cmpi eq, %rem, %c0 : index
      scf.if %is_print {
        func.call @print_f32(%loss) : (f32) -> ()
        func.call @print_newline() : () -> ()
      }

      func.call @compute_gradients(%X, %W, %b, %y_true, %dW, %db) : (memref<4x2xf32>, memref<2x1xf32>, memref<1xf32>, memref<4x1xf32>, memref<2x1xf32>, memref<1xf32>) -> ()
      func.call @sgd_update(%W, %b, %dW, %db, %lr) : (memref<2x1xf32>, memref<1xf32>, memref<2x1xf32>, memref<1xf32>, f32) -> ()
    }

    // 打印最终结果
    %final_loss = func.call @forward_loss(%X, %W, %b, %y_true) : (memref<4x2xf32>, memref<2x1xf32>, memref<1xf32>, memref<4x1xf32>) -> f32
    func.call @print_f32(%final_loss) : (f32) -> ()
    func.call @print_newline() : () -> ()

    %w0 = memref.load %W[%i0, %i0] : memref<2x1xf32>
    %w1 = memref.load %W[%i1, %i0] : memref<2x1xf32>
    %b0 = memref.load %b[%i0] : memref<1xf32>
    func.call @print_newline() : () -> ()
    func.call @print_f32(%w0) : (f32) -> ()
    func.call @print_newline() : () -> ()
    func.call @print_f32(%w1) : (f32) -> ()
    func.call @print_newline() : () -> ()
    func.call @print_f32(%b0) : (f32) -> ()
    func.call @print_newline() : () -> ()

    return
  }
}
