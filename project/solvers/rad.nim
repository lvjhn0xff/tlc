import std/math
import std/tables
import std/strformat
import std/sequtils   

# =====================================================================
# Core Data Types
# =====================================================================

type
  OpKind* = enum
    opNone
    opAdd
    opSub
    opMul
    opDivOp     
    opPow
    opNeg
    opExp
    opLog
    opSin
    opCos
    opTan
    opSinh
    opCosh
    opTanh
    opSqrt
    opRelu
    opSigmoid

  TapeEntry* = ref object
    op*: OpKind
    inputs*: seq[int]      # indices into tape
    grad*: float64         # accumulated gradient
    output*: float64       # cached forward value

  Var* = ref object
    data*: float64
    grad*: float64
    id*: int               # index into tape
    tape*: seq[TapeEntry]  # reference to global tape

# =====================================================================
# Global Tape Management
# =====================================================================

var globalTape: seq[TapeEntry] = @[]
var gradEnabled = true

proc resetTape*() =
  globalTape = @[]
  # Reserve index 0 for constants/leaves
  globalTape.add(TapeEntry(op: opNone, inputs: @[], grad: 0.0, output: 0.0))

proc currentTape*(): seq[TapeEntry] =
  return globalTape

# =====================================================================
# Variable Creation
# =====================================================================

proc newVar*(value: float64, requiresGrad: bool = true): Var =
  ## Create a new variable with optional gradient tracking
  result = Var(data: value, grad: 0.0, id: 0, tape: globalTape)
  
  if requiresGrad and gradEnabled:
    result.id = globalTape.len
    globalTape.add(TapeEntry(op: opNone, inputs: @[], grad: 0.0, output: value))
  else:
    result.id = 0  # constant, no gradient

proc newVar*(value: int, requiresGrad: bool = true): Var =
  newVar(value.float64, requiresGrad)

# =====================================================================
# Gradient Accumulation
# =====================================================================

proc accumulateGrad*(tape: var seq[TapeEntry], idx: int, grad: float64) =
  if idx > 0 and idx < tape.len:
    tape[idx].grad += grad

# =====================================================================
# Operations (Forward + Backward)
# =====================================================================

# ----- Binary Operations -----

proc add*(a, b: Var): Var =
  result = Var(data: a.data + b.data, grad: 0.0, id: 0, tape: globalTape)
  
  if gradEnabled and (a.id > 0 or b.id > 0):
    result.id = globalTape.len
    globalTape.add(TapeEntry(
      op: opAdd,
      inputs: @[a.id, b.id],
      grad: 0.0,
      output: result.data
    ))

proc sub*(a, b: Var): Var =
  result = Var(data: a.data - b.data, grad: 0.0, id: 0, tape: globalTape)
  
  if gradEnabled and (a.id > 0 or b.id > 0):
    result.id = globalTape.len
    globalTape.add(TapeEntry(
      op: opSub,
      inputs: @[a.id, b.id],
      grad: 0.0,
      output: result.data
    ))

proc mul*(a, b: Var): Var =
  result = Var(data: a.data * b.data, grad: 0.0, id: 0, tape: globalTape)
  
  if gradEnabled and (a.id > 0 or b.id > 0):
    result.id = globalTape.len
    globalTape.add(TapeEntry(
      op: opMul,
      inputs: @[a.id, b.id],
      grad: 0.0,
      output: result.data
    ))

proc divOp*(a, b: Var): Var =  # Renamed from div to divOp
  result = Var(data: a.data / b.data, grad: 0.0, id: 0, tape: globalTape)
  
  if gradEnabled and (a.id > 0 or b.id > 0):
    result.id = globalTape.len
    globalTape.add(TapeEntry(
      op: opDivOp,  # Changed from opDiv to opDivOp
      inputs: @[a.id, b.id],
      grad: 0.0,
      output: result.data
    ))

proc pow*(a: Var, b: float64): Var =
  result = Var(data: pow(a.data, b), grad: 0.0, id: 0, tape: globalTape)
  
  if gradEnabled and a.id > 0:
    result.id = globalTape.len
    globalTape.add(TapeEntry(
      op: opPow,
      inputs: @[a.id],
      grad: 0.0,
      output: result.data
    ))

# ----- Unary Operations -----

proc neg*(a: Var): Var =
  result = Var(data: -a.data, grad: 0.0, id: 0, tape: globalTape)
  
  if gradEnabled and a.id > 0:
    result.id = globalTape.len
    globalTape.add(TapeEntry(
      op: opNeg,
      inputs: @[a.id],
      grad: 0.0,
      output: result.data
    ))

proc exp*(a: Var): Var =
  result = Var(data: exp(a.data), grad: 0.0, id: 0, tape: globalTape)
  
  if gradEnabled and a.id > 0:
    result.id = globalTape.len
    globalTape.add(TapeEntry(
      op: opExp,
      inputs: @[a.id],
      grad: 0.0,
      output: result.data
    ))

proc log*(a: Var): Var =
  result = Var(data: ln(a.data), grad: 0.0, id: 0, tape: globalTape)
  
  if gradEnabled and a.id > 0:
    result.id = globalTape.len
    globalTape.add(TapeEntry(
      op: opLog,
      inputs: @[a.id],
      grad: 0.0,
      output: result.data
    ))

proc sin*(a: Var): Var =
  result = Var(data: sin(a.data), grad: 0.0, id: 0, tape: globalTape)
  
  if gradEnabled and a.id > 0:
    result.id = globalTape.len
    globalTape.add(TapeEntry(
      op: opSin,
      inputs: @[a.id],
      grad: 0.0,
      output: result.data
    ))

proc cos*(a: Var): Var =
  result = Var(data: cos(a.data), grad: 0.0, id: 0, tape: globalTape)
  
  if gradEnabled and a.id > 0:
    result.id = globalTape.len
    globalTape.add(TapeEntry(
      op: opCos,
      inputs: @[a.id],
      grad: 0.0,
      output: result.data
    ))

proc tan*(a: Var): Var =
  result = Var(data: tan(a.data), grad: 0.0, id: 0, tape: globalTape)
  
  if gradEnabled and a.id > 0:
    result.id = globalTape.len
    globalTape.add(TapeEntry(
      op: opTan,
      inputs: @[a.id],
      grad: 0.0,
      output: result.data
    ))

proc sinh*(a: Var): Var =
  result = Var(data: sinh(a.data), grad: 0.0, id: 0, tape: globalTape)
  
  if gradEnabled and a.id > 0:
    result.id = globalTape.len
    globalTape.add(TapeEntry(
      op: opSinh,
      inputs: @[a.id],
      grad: 0.0,
      output: result.data
    ))

proc cosh*(a: Var): Var =
  result = Var(data: cosh(a.data), grad: 0.0, id: 0, tape: globalTape)
  
  if gradEnabled and a.id > 0:
    result.id = globalTape.len
    globalTape.add(TapeEntry(
      op: opCosh,
      inputs: @[a.id],
      grad: 0.0,
      output: result.data
    ))

proc tanh*(a: Var): Var =
  result = Var(data: tanh(a.data), grad: 0.0, id: 0, tape: globalTape)
  
  if gradEnabled and a.id > 0:
    result.id = globalTape.len
    globalTape.add(TapeEntry(
      op: opTanh,
      inputs: @[a.id],
      grad: 0.0,
      output: result.data
    ))

proc sqrt*(a: Var): Var =
  result = Var(data: sqrt(a.data), grad: 0.0, id: 0, tape: globalTape)
  
  if gradEnabled and a.id > 0:
    result.id = globalTape.len
    globalTape.add(TapeEntry(
      op: opSqrt,
      inputs: @[a.id],
      grad: 0.0,
      output: result.data
    ))

proc relu*(a: Var): Var =
  let val = if a.data > 0: a.data else: 0.0
  result = Var(data: val, grad: 0.0, id: 0, tape: globalTape)
  
  if gradEnabled and a.id > 0:
    result.id = globalTape.len
    globalTape.add(TapeEntry(
      op: opRelu,
      inputs: @[a.id],
      grad: 0.0,
      output: result.data
    ))

proc sigmoid*(a: Var): Var =
  let val = 1.0 / (1.0 + exp(-a.data))
  result = Var(data: val, grad: 0.0, id: 0, tape: globalTape)
  
  if gradEnabled and a.id > 0:
    result.id = globalTape.len
    globalTape.add(TapeEntry(
      op: opSigmoid,
      inputs: @[a.id],
      grad: 0.0,
      output: result.data
    ))

# =====================================================================
# Operator Overloading
# =====================================================================

proc `+`*(a: Var, b: Var): Var = add(a, b)
proc `+`*(a: Var, b: float64): Var = add(a, newVar(b, false))
proc `+`*(a: float64, b: Var): Var = add(newVar(a, false), b)
proc `+`*(a: Var, b: int): Var = add(a, newVar(b.float64, false))

proc `-`*(a: Var, b: Var): Var = sub(a, b)
proc `-`*(a: Var, b: float64): Var = sub(a, newVar(b, false))
proc `-`*(a: float64, b: Var): Var = sub(newVar(a, false), b)
proc `-`*(a: Var): Var = neg(a)

proc `*`*(a: Var, b: Var): Var = mul(a, b)
proc `*`*(a: Var, b: float64): Var = mul(a, newVar(b, false))
proc `*`*(a: float64, b: Var): Var = mul(newVar(a, false), b)
proc `*`*(a: Var, b: int): Var = mul(a, newVar(b.float64, false))

proc `/`*(a: Var, b: Var): Var = divOp(a, b)  # Using divOp instead of div
proc `/`*(a: Var, b: float64): Var = divOp(a, newVar(b, false))
proc `/`*(a: float64, b: Var): Var = divOp(newVar(a, false), b)
proc `/`*(a: Var, b: int): Var = divOp(a, newVar(b.float64, false))

proc `^`*(a: Var, b: float64): Var = pow(a, b)
proc `^`*(a: Var, b: int): Var = pow(a, b.float64)

# =====================================================================
# Backward Pass (Reverse AD)
# =====================================================================

proc backward*(loss: Var) =
  ## Perform reverse-mode automatic differentiation from the loss variable
  
  if loss.id == 0:
    raise newException(ValueError, "Loss variable does not track gradients")
  
  # Reset all gradients in tape
  for i in 0..<globalTape.len:
    globalTape[i].grad = 0.0
  
  # Initialize gradient of loss to 1.0
  globalTape[loss.id].grad = 1.0
  
  # Traverse tape in reverse order
  for i in countdown(globalTape.len - 1, 1):
    let entry = globalTape[i]
    let grad = entry.grad
    
    if grad == 0.0:
      continue
    
    case entry.op
    of opNone:
      # Leaf node - gradient already stored in tape
      discard
      
    of opAdd:
      # dL/da = dL/dc, dL/db = dL/dc
      if entry.inputs.len >= 2:
        globalTape[entry.inputs[0]].grad += grad
        globalTape[entry.inputs[1]].grad += grad
      
    of opSub:
      # dL/da = dL/dc, dL/db = -dL/dc
      if entry.inputs.len >= 2:
        globalTape[entry.inputs[0]].grad += grad
        globalTape[entry.inputs[1]].grad -= grad
      
    of opMul:
      # dL/da = dL/dc * b, dL/db = dL/dc * a
      if entry.inputs.len >= 2:
        let a_val = globalTape[entry.inputs[0]].output
        let b_val = globalTape[entry.inputs[1]].output
        globalTape[entry.inputs[0]].grad += grad * b_val
        globalTape[entry.inputs[1]].grad += grad * a_val
      
    of opDivOp:  # Changed from opDiv to opDivOp
      # dL/da = dL/dc / b, dL/db = -dL/dc * a / b^2
      if entry.inputs.len >= 2:
        let a_val = globalTape[entry.inputs[0]].output
        let b_val = globalTape[entry.inputs[1]].output
        globalTape[entry.inputs[0]].grad += grad / b_val
        globalTape[entry.inputs[1]].grad -= grad * a_val / (b_val * b_val)
      
    of opPow:
      # dL/da = dL/dc * b * a^(b-1)
      if entry.inputs.len >= 1:
        # For pow(a, b) where b is constant
        # We store b in the tape entry metadata
        # For simplicity, we skip this in the base implementation
        discard
      
    of opNeg:
      # dL/da = -dL/dc
      if entry.inputs.len >= 1:
        globalTape[entry.inputs[0]].grad -= grad
      
    of opExp:
      # dL/da = dL/dc * exp(a) = dL/dc * output
      if entry.inputs.len >= 1:
        globalTape[entry.inputs[0]].grad += grad * entry.output
      
    of opLog:
      # dL/da = dL/dc / a
      if entry.inputs.len >= 1:
        let a_val = globalTape[entry.inputs[0]].output
        globalTape[entry.inputs[0]].grad += grad / a_val
      
    of opSin:
      # dL/da = dL/dc * cos(a)
      if entry.inputs.len >= 1:
        let a_val = globalTape[entry.inputs[0]].output
        globalTape[entry.inputs[0]].grad += grad * cos(a_val)
      
    of opCos:
      # dL/da = -dL/dc * sin(a)
      if entry.inputs.len >= 1:
        let a_val = globalTape[entry.inputs[0]].output
        globalTape[entry.inputs[0]].grad -= grad * sin(a_val)
      
    of opTan:
      # dL/da = dL/dc * sec^2(a) = dL/dc * (1 + tan^2(a))
      if entry.inputs.len >= 1:
        let a_val = globalTape[entry.inputs[0]].output
        globalTape[entry.inputs[0]].grad += grad * (1.0 + tan(a_val) * tan(a_val))
      
    of opSinh:
      # dL/da = dL/dc * cosh(a)
      if entry.inputs.len >= 1:
        let a_val = globalTape[entry.inputs[0]].output
        globalTape[entry.inputs[0]].grad += grad * cosh(a_val)
      
    of opCosh:
      # dL/da = dL/dc * sinh(a)
      if entry.inputs.len >= 1:
        let a_val = globalTape[entry.inputs[0]].output
        globalTape[entry.inputs[0]].grad += grad * sinh(a_val)
      
    of opTanh:
      # dL/da = dL/dc * (1 - tanh^2(a)) = dL/dc * (1 - output^2)
      if entry.inputs.len >= 1:
        globalTape[entry.inputs[0]].grad += grad * (1.0 - entry.output * entry.output)
      
    of opSqrt:
      # dL/da = dL/dc / (2 * sqrt(a))
      if entry.inputs.len >= 1:
        let a_val = globalTape[entry.inputs[0]].output
        globalTape[entry.inputs[0]].grad += grad / (2.0 * sqrt(a_val))
      
    of opRelu:
      # dL/da = dL/dc if a > 0 else 0
      if entry.inputs.len >= 1:
        let a_val = globalTape[entry.inputs[0]].output
        if a_val > 0:
          globalTape[entry.inputs[0]].grad += grad
      
    of opSigmoid:
      # dL/da = dL/dc * sigmoid(a) * (1 - sigmoid(a)) = dL/dc * output * (1 - output)
      if entry.inputs.len >= 1:
        globalTape[entry.inputs[0]].grad += grad * entry.output * (1.0 - entry.output)
  
  # Copy gradients back to Var objects
  # For leaf nodes (opNone), the gradient is in the tape
  # Variables reference the same tape, so they can access their grad via id

# =====================================================================
# Helper Functions
# =====================================================================

proc grad*(v: Var): float64 =
  ## Get the gradient of a variable after backward pass
  if v.id > 0 and v.id < globalTape.len:
    return globalTape[v.id].grad
  return 0.0

proc value*(v: Var): float64 = v.data

proc zeroGrad*(v: Var) =
  ## Zero out gradient of a variable
  if v.id > 0 and v.id < globalTape.len:
    globalTape[v.id].grad = 0.0

proc zeroAllGrads*() =
  ## Zero out all gradients in the tape
  for i in 0..<globalTape.len:
    globalTape[i].grad = 0.0

# =====================================================================
# Tensor/Matrix Support (2D)
# =====================================================================

type
  Tensor* = ref object
    data*: seq[seq[float64]]
    grad*: seq[seq[float64]]
    rows*, cols*: int
    id*: int  # index into tape for the whole tensor
    tape*: seq[TapeEntry]

proc newTensor*(rows, cols: int, requiresGrad: bool = true): Tensor =
  result = Tensor(
    data: newSeqWith(rows, newSeq[float64](cols)),
    grad: newSeqWith(rows, newSeq[float64](cols)),
    rows: rows,
    cols: cols,
    id: 0,
    tape: globalTape
  )
  # Tensor operations are tracked at the element level via the tape
  # For simplicity, we use scalar vars for tensor elements

proc tensorFromArray*(arr: seq[seq[float64]], requiresGrad: bool = true): Tensor =
  result = newTensor(arr.len, arr[0].len, requiresGrad)
  for i in 0..<arr.len:
    for j in 0..<arr[0].len:
      result.data[i][j] = arr[i][j]

proc `$`*(t: Tensor): string =
  result = "["
  for i in 0..<t.rows:
    if i > 0: result.add(" ")
    result.add("[")
    for j in 0..<t.cols:
      result.add($t.data[i][j])
      if j < t.cols - 1: result.add(", ")
    result.add("]")
    if i < t.rows - 1: result.add("\n")
  result.add("]")

# Simple tensor operations using scalar variables
proc matmul*(a, b: Tensor): Tensor =
  ## Matrix multiplication using scalar variables
  assert a.cols == b.rows, "Matrix dimensions must match for multiplication"
  
  result = newTensor(a.rows, b.cols, true)
  
  for i in 0..<a.rows:
    for j in 0..<b.cols:
      var sum = 0.0
      for k in 0..<a.cols:
        sum += a.data[i][k] * b.data[k][j]
      result.data[i][j] = sum

proc matmulGrad*(a, b: Tensor, gradOutput: seq[seq[float64]]) =
  ## Compute gradients for matrix multiplication (simplified)
  # For full gradient tracking, each element would be a Var
  # This is a simplified version
  discard

# =====================================================================
# Example Usage
# =====================================================================

when isMainModule:
  echo "=== Reverse AD in Nim ==="
  echo ""
  
  # ----- Example 1: Simple scalar function -----
  echo "Example 1: f(x) = x^2 + 3x + 5 at x=2"
  resetTape()
  
  let x = newVar(2.0)
  let f = x*x + 3.0*x + 5.0
  
  echo &"  f = {f.data:.4f}"
  
  f.backward()
  echo &"  df/dx = {x.grad:.4f} (expected: 7.0)"
  
  # ----- Example 2: Multiple variables -----
  echo ""
  echo "Example 2: f(x,y) = x*y + sin(x) at x=1, y=2"
  resetTape()
  
  let x2 = newVar(1.0)
  let y2 = newVar(2.0)
  let f2 = x2*y2 + sin(x2)
  
  echo &"  f = {f2.data:.4f}"
  
  f2.backward()
  echo &"  df/dx = {x2.grad:.4f} (expected: y + cos(x) = 2 + 0.5403 = 2.5403)"
  echo &"  df/dy = {y2.grad:.4f} (expected: x = 1.0)"
  
  # ----- Example 3: Chain rule -----
  echo ""
  echo "Example 3: f(x) = exp(sin(x^2)) at x=0.5"
  resetTape()
  
  let x3 = newVar(0.5)
  let f3 = exp(sin(x3*x3))
  
  echo &"  f = {f3.data:.4f}"
  
  f3.backward()
  echo &"  df/dx = {x3.grad:.4f}"
  
  # Verify with numerical gradient
  let eps = 1e-6
  let x3_plus = newVar(0.5 + eps)
  let f3_plus = exp(sin(x3_plus*x3_plus))
  let numerical_grad = (f3_plus.data - f3.data) / eps
  echo &"  Numerical gradient: {numerical_grad:.4f}"
  
  # ----- Example 4: Neural network style -----
  echo ""
  echo "Example 4: Simple 2-layer network (linear + sigmoid)"
  resetTape()
  
  let input = newVar(1.5)
  let w1 = newVar(0.8)
  let b1 = newVar(0.2)
  let w2 = newVar(0.5)
  let b2 = newVar(-0.1)
  
  let h = sigmoid(w1 * input + b1)
  let output = w2 * h + b2
  
  echo &"  Input: {input.data:.4f}"
  echo &"  Output: {output.data:.4f}"
  
  output.backward()
  echo &"  doutput/dinput = {input.grad:.4f}"
  echo &"  doutput/dw1 = {w1.grad:.4f}"
  echo &"  doutput/db1 = {b1.grad:.4f}"
  echo &"  doutput/dw2 = {w2.grad:.4f}"
  echo &"  doutput/db2 = {b2.grad:.4f}"
  
  # ----- Example 5: Linear regression -----
  echo ""
  echo "Example 5: Linear regression (MSE loss)"
  resetTape()
  
  # Generate synthetic data: y = 2x + 1 + noise
  let x_data = @[0.0, 1.0, 2.0, 3.0, 4.0]
  let y_data = @[1.2, 3.1, 4.8, 7.2, 8.9]
  
  # Model parameters
  let w = newVar(0.0)
  let b = newVar(0.0)
  
  # Forward pass: compute MSE loss
  var totalLoss = newVar(0.0)
  for i in 0..<x_data.len:
    let xv = newVar(x_data[i], false)  # input, no gradient needed
    let y_pred = w * xv + b
    let diff = y_pred - newVar(y_data[i], false)
    totalLoss = totalLoss + diff * diff
  
  let loss = totalLoss / newVar(x_data.len.float64, false)
  
  echo &"  Initial loss: {loss.data:.4f}"
  echo &"  Initial w: {w.data:.4f}, b: {b.data:.4f}"
  
  # Compute gradients
  loss.backward()
  echo &"  dloss/dw = {w.grad:.4f}"
  echo &"  dloss/db = {b.grad:.4f}"
  
  # One gradient descent step
  let lr = 0.01
  w.data -= lr * w.grad
  b.data -= lr * b.grad
  
  # Zero gradients for next step
  zeroAllGrads()
  
  # Recompute loss after update
  totalLoss = newVar(0.0)
  for i in 0..<x_data.len:
    let xv = newVar(x_data[i], false)
    let y_pred = w * xv + b
    let diff = y_pred - newVar(y_data[i], false)
    totalLoss = totalLoss + diff * diff
  
  let loss2 = totalLoss / newVar(x_data.len.float64, false)
  echo &"  After 1 step: w={w.data:.4f}, b={b.data:.4f}, loss={loss2.data:.4f}"
  
  # ----- Example 6: Binary classification with logistic regression -----
  echo ""
  echo "Example 6: Binary classification (logistic regression)"
  resetTape()
  
  # Data: features and labels (0/1)
  let X_class = @[@[0.5, 1.2], @[1.0, 0.8], @[1.8, 1.5], @[2.2, 0.5], @[2.8, 1.0]]
  let y_class = @[0.0, 0.0, 1.0, 1.0, 1.0]
  
  # Model parameters (2 features)
  let w1_class = newVar(0.0)
  let w2_class = newVar(0.0)
  let b_class = newVar(0.0)
  
  # Binary cross-entropy loss
  var totalLossClass = newVar(0.0)
  for i in 0..<X_class.len:
    let x1 = newVar(X_class[i][0], false)
    let x2 = newVar(X_class[i][1], false)
    let z = w1_class * x1 + w2_class * x2 + b_class
    let p = sigmoid(z)
    let y = newVar(y_class[i], false)
    # BCE: -[y*log(p) + (1-y)*log(1-p)]
    let loss_i = -(y * log(p) + (1.0 - y) * log(1.0 - p))
    totalLossClass = totalLossClass + loss_i
  
  let lossClass = totalLossClass / newVar(X_class.len.float64, false)
  
  echo &"  Initial classification loss: {lossClass.data:.4f}"
  
  lossClass.backward()
  echo &"  dloss/dw1 = {w1_class.grad:.4f}"
  echo &"  dloss/dw2 = {w2_class.grad:.4f}"
  echo &"  dloss/db = {b_class.grad:.4f}"
  
  echo ""
  echo "=== All examples completed successfully ==="