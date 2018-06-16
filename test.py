import numpy as np
from layers import *
from ann import *

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
  """ 
  a naive implementation of numerical gradient of f at x 
  - f should be a function that takes a single argument
  - x is the point (numpy array) to evaluate the gradient at
  """ 

  fx = f(x) # evaluate function value at original point
  grad = np.zeros_like(x)
  # iterate over all indexes in x
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:

    # evaluate function at x+h
    ix = it.multi_index
    oldval = x[ix]
    x[ix] = oldval + h # increment by h
    fxph = f(x) # evalute f(x + h)
    x[ix] = oldval - h
    fxmh = f(x) # evaluate f(x - h)
    x[ix] = oldval # restore

    # compute the partial derivative with centered formula
    grad[ix] = (fxph - fxmh) / (2 * h) # the slope
    if verbose:
      print ix, grad[ix]
    it.iternext() # step to next dimension

  return grad

def eval_numerical_gradient_array(f, x, df, h=1e-5):
  """
  Evaluate a numeric gradient for a function that accepts a numpy
  array and returns a numpy array.
  """
  grad = np.zeros_like(x)
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:
    ix = it.multi_index
    
    oldval = x[ix]
    x[ix] = oldval + h
    pos = f(x).copy()
    x[ix] = oldval - h
    neg = f(x).copy()
    x[ix] = oldval
    
    grad[ix] = np.sum((pos - neg) * df) / (2 * h)
    it.iternext()
  return grad

def test_affine_forward():
	num_inputs = 2
	input_dim = 4
	output_dim = 3

	input_size = num_inputs * input_dim
	weight_size = output_dim * input_dim

	x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, input_dim)
	w = np.linspace(-0.2, 0.3, num=weight_size).reshape(input_dim, output_dim)
	b = np.linspace(-0.3, 0.1, num=output_dim)

	out, _ = affine_forward(x, w, b)
	correct_out = np.array([[-0.24103896, -0.03584416,  0.16935065],
 							[-0.23480519,  0.03272727,  0.30025974]])

	# Compare your output with ours. The error should be around 1e-9.
	print 'Testing affine_forward function:'
	print 'difference: ', rel_error(out, correct_out)
	print
	
def test_affine_backward():
	x = np.random.randn(10, 6)
	w = np.random.randn(6, 5)
	b = np.random.randn(5)
	dout = np.random.randn(10, 5)

	dx_num = eval_numerical_gradient_array(lambda x: affine_forward(x, w, b)[0], x, dout)
	dw_num = eval_numerical_gradient_array(lambda w: affine_forward(x, w, b)[0], w, dout)
	db_num = eval_numerical_gradient_array(lambda b: affine_forward(x, w, b)[0], b, dout)

	_, cache = affine_forward(x, w, b)
	dx, dw, db = affine_backward(dout, cache)

	# The error should be around 1e-10
	print 'Testing affine_backward function:'
	print 'dx error: ', rel_error(dx_num, dx)
	print 'dw error: ', rel_error(dw_num, dw)
	print 'db error: ', rel_error(db_num, db)
	print

def test_relu_forward():
	x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)

	out, _ = relu_forward(x)
	correct_out = np.array([[ 0.,          0.,          0.,          0.,        ],
		                    [ 0.,          0.,          0.04545455,  0.13636364,],
		                    [ 0.22727273,  0.31818182,  0.40909091,  0.5,       ]])

	# Compare your output with ours. The error should be around 1e-8
	print 'Testing relu_forward function:'
	print 'difference: ', rel_error(out, correct_out)
	print
	
def test_relu_backward():
	x = np.random.randn(10, 10)
	dout = np.random.randn(*x.shape)

	dx_num = eval_numerical_gradient_array(lambda x: relu_forward(x)[0], x, dout)

	_, cache = relu_forward(x)
	dx = relu_backward(dout, cache)

	# The error should be around 1e-12
	print 'Testing relu_backward function:'
	print 'dx error: ', rel_error(dx_num, dx)
	print

def test_L2_loss():
	x = np.array([3.5,1.2,4.0])
	y = np.array([3.3,1.4,4.1])
	h = 0.00001
	correct_out = 0.015

	correct_dx = np.array([0.066666667, -0.066666667, -0.033333334])
	loss, dx = L2_loss(x, y)


	# The error should be around 1e-12
	print 'Testing L2_loss function:'
	print 'loss error: ', rel_error(correct_out, loss)
	print 'dx error: ', rel_error(correct_dx, dx)
	print 
	
def test_ANN_predict():
	net = ANN([3],2)

	net.params['b0'][:] = (np.arange(3,dtype=np.float64)+3.).reshape(net.params['b0'].shape)
	net.params['b1'][:] = (np.arange(1,dtype=np.float64)+4.).reshape(net.params['b1'].shape)

	net.params['W0'] = (np.arange(6,dtype=np.float64)+1.).reshape(net.params['W0'].shape)
	net.params['W1'] = (np.arange(3,dtype=np.float64)+7.).reshape(net.params['W1'].shape)

	x = np.array([[1., 2.],[3., 4.],[5.,6.]])
	y = np.array([[396.], [740.], [1084.]])

	y_hat = net.predict(x)
	print 'Testing ANN.predict function:'
	print 'prediction error:',rel_error(y,y_hat)
