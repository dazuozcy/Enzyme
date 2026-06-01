// Affine maps for linalg operations
#map_broadcast_add = affine_map<(d0, d1) -> (d0, d1)>
#map_broadcast_add_b = affine_map<(d0, d1) -> (d1)>
#map_sub = affine_map<(d0, d1) -> (d0, d1)>
#map_square = affine_map<(d0, d1) -> (d0, d1)>

// ============================================================================
// Forward Pass: Linear layer y = X @ W + b
// ============================================================================
func.func @forward(%X: tensor<4x2xf32>, %W: tensor<2x1xf32>, %b: tensor<1xf32>)
  -> tensor<4x1xf32> {
  %cst_zero = arith.constant 0.0 : f32
  %zero_4x1 = arith.constant dense<0.0> : tensor<4x1xf32>

  // Matrix multiplication: X @ W
  %matmul = linalg.matmul ins(%X, %W : tensor<4x2xf32>, tensor<2x1xf32>)
    outs(%zero_4x1 : tensor<4x1xf32>) -> tensor<4x1xf32>

  // Add bias: matmul + b (broadcast)
  %result = linalg.generic
    {indexing_maps = [#map_broadcast_add, #map_broadcast_add_b, #map_broadcast_add],
     iterator_types = ["parallel", "parallel"]}
    ins(%matmul, %b : tensor<4x1xf32>, tensor<1xf32>) outs(%zero_4x1 : tensor<4x1xf32>) {
  ^bb0(%in1: f32, %in2: f32, %out: f32):
    %sum = arith.addf %in1, %in2 : f32
    linalg.yield %sum : f32
  } -> tensor<4x1xf32>

  return %result : tensor<4x1xf32>
}

// ============================================================================
// Loss Function: MSE = mean((y_pred - y_true)^2)
// ============================================================================
func.func @mse_loss(%y_pred: tensor<4x1xf32>, %y_true: tensor<4x1xf32>) -> f32 {
  %cst_zero = arith.constant 0.0 : f32
  %zero_4x1 = arith.constant dense<0.0> : tensor<4x1xf32>
  %cst_4 = arith.constant 4.0 : f32

  // Compute difference: y_pred - y_true
  %diff = arith.subf %y_pred, %y_true : tensor<4x1xf32>
  %squared = arith.mulf %diff, %diff : tensor<4x1xf32>

  // Sum all elements - use tensor<f32> for reduction
  %zero_scalar = arith.constant dense<0.0> : tensor<f32>
  %sum_tensor = linalg.reduce ins(%squared : tensor<4x1xf32>) outs(%zero_scalar : tensor<f32>) dimensions = [0, 1]
                (%in: f32, %out: f32) { %sum = arith.addf %in, %out : f32 linalg.yield %sum : f32}

  // Extract scalar from tensor
  %sum = tensor.extract %sum_tensor[] : tensor<f32>

  // Divide by number of samples (4)
  %mse = arith.divf %sum, %cst_4 : f32

  return %mse : f32
}


// ============================================================================
// SGD Optimizer: W = W - lr * dW, b = b - lr * db
// ============================================================================
func.func @sgd_update(
  %W: tensor<2x1xf32>, %b: tensor<1xf32>, 
  %dW: tensor<2x1xf32>, %db: tensor<1xf32>, %lr: f32
) -> (tensor<2x1xf32>, tensor<1xf32>) attributes {no_inline} {
  %zero_2x1 = arith.constant dense<0.0> : tensor<2x1xf32>
  %zero_1 = arith.constant dense<0.0> : tensor<1xf32>

  // Update W: W_new = W - lr * dW
  %W_new = linalg.generic
    {indexing_maps = [#map_broadcast_add, #map_broadcast_add, #map_broadcast_add],
     iterator_types = ["parallel", "parallel"]}
    ins(%W, %dW : tensor<2x1xf32>, tensor<2x1xf32>)
    outs(%zero_2x1 : tensor<2x1xf32>) {
  ^bb0(%w: f32, %dw: f32, %out: f32):
    %scaled = arith.mulf %lr, %dw : f32
    %updated = arith.subf %w, %scaled : f32
    linalg.yield %updated : f32
  } -> tensor<2x1xf32>

  // Update b: b_new = b - lr * db
  %b_new = linalg.generic
    {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
     iterator_types = ["parallel"]}
    ins(%b, %db : tensor<1xf32>, tensor<1xf32>)
    outs(%zero_1 : tensor<1xf32>) {
  ^bb0(%bi: f32, %dbi: f32, %out: f32):
    %scaled = arith.mulf %lr, %dbi : f32
    %updated = arith.subf %bi, %scaled : f32
    linalg.yield %updated : f32
  } -> tensor<1xf32>

  return %W_new, %b_new : tensor<2x1xf32>, tensor<1xf32>
}
