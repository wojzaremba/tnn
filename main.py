#!/usr/bin/python
import os
import theano
from theano import tensor as T
import numpy as np
import copy
import math

W = []
dW = []
GC = False

if GC:
  hidden = 3
else:
  hidden = 50
lr = 0.01


def arity(data):
  s = set()
  if type(data) is list:
    for d in data:
      s = s.union(arity(d))
  elif data is not None:
    if len(data) == 2:
      root, expr1 = data
      s.add((root, 2))
      s = s.union(arity(expr1))
    else:
      root, expr1, expr2 = data
      s.add((root, 3))
      s = s.union(arity(expr1))
      s = s.union(arity(expr2))
  return s

def nonlin(x):
  assert(len(x[0].shape) == 1)
  return np.tanh(x[0]), x
  #return np.maximum(x[0], 0), x

def dnonlin((dX_in, x)):
  assert(dX_in.shape == x[0].shape)
  assert(len(x[0].shape) == 1)
  return dX_in * (1 - np.square(x[0])), x[1]
  #return dX_in * (x[0] > 0), x[1]

def tnn_fp(x):
  global W
  if x is None:
    return (W[-1],)
  elif len(x) == 2:
    root, expr0 = x
    f0 = tnn_fp(expr0)
    return nonlin((np.dot(W[root], f0[0]), f0))
  elif len(x) == 3:
    root, expr0, expr1 = x
    f0 = tnn_fp(expr0)
    f1 = tnn_fp(expr1)
    f = np.dot(np.tensordot(W[root], f0[0], axes=(2, 0)), f1[0])
    return nonlin((f, f0, f1))

def tnn_bp((dX_in, x), data):
  global dW
  if data is None:
    return
  dX_in, x = dnonlin((dX_in, x))
  if len(data) == 2:
    assert(len(x) == 2)
    dW[data[0]] += np.outer(dX_in, x[1][0])
    dX = np.dot(W[data[0]].transpose(), dX_in)
    tnn_bp((dX, x[1]), data[1])
  elif len(data) == 3:
    assert(len(x) == 3)
    dX_r = dX_in.reshape(list(dX_in.shape) + [1])
    o = np.outer(x[2][0], x[1][0])
    o = o.reshape(list(o.shape) + [1])
    dW[data[0]] += np.tensordot(dX_r, o, axes=(1, 2))
    dX0 = np.dot(np.tensordot(W[data[0]], x[2][0], axes=(1, 0)).transpose(), dX_in)
    tnn_bp((dX0, x[1]), data[1])
    dX1 = np.dot(np.tensordot(W[data[0]], x[1][0], axes=(2, 0)).transpose(), dX_in)
    tnn_bp((dX1, x[2]), data[2])

def lin_fp(x):
  global W
  return np.dot(W[-2], x[0]), x

def lin_bp((dX_in, x)):
  global W, dW
  dX = np.dot(W[-2].transpose(), dX_in)
  dW[-1] = np.outer(dX_in, x[1][0])
  return dX, x[1]

def softmax_fp(x):
  output = x[0] - np.max(x[0])
  output = np.exp(output)
  output = output / np.sum(output)
  pred = np.argmax(output)
  return (output, x), pred

def softmax_bp(x, y):
  dX = x[0]
  dX[y] -= 1
  return dX, x[1]

def init():
  global W, dW, hidden
  np.random.seed(1)
  path = "training/"
  files = os.listdir(path)
  data = []
  for d in files:
    same = []
    with open(path + d, 'r') as f:
      lines = f.readlines()
      for l in lines:
        l = eval(l)
        same.append(l)
    if len(same) > 2:
      data.append(same)
  classes = len(data)
  ar = sorted(arity(data))
  for i in xrange(len(ar)):
    s = [hidden] * ar[i][1]
    print "Size of %d is %s" % (i, str(s))
    W.append(np.random.randn(*s) * 0.001)
    dW.append(np.zeros(tuple(s)))
  W.append(np.random.randn(classes, hidden))
  dW.append(np.zeros_like(W[-1]))
  W.append(np.ones(hidden))
  return data

def fp(d):
  return softmax_fp(lin_fp(tnn_fp(d)))

def bp(output, scratch, y, data):
  tnn_bp(lin_bp(softmax_bp((output, scratch), y)), data)

def update_weights():
  global W, lr, dW
  for i in xrange(len(dW)):
    W[i] = W[i] - lr * dW[i] 

def norm(x):
  return np.sqrt(np.sum(np.square(x)))

def assert_close(a, b):
  if norm(a) < 1e-5 and norm(b) < 1e-5:
    return
  diff = norm(a - b.reshape(a.shape)) / norm(b) 
  if diff > 1e-4 or math.isnan(diff):
    print "diff = %f, norm(a) = %f, norm(b) = %f" % (diff, norm(a), norm(b))
    a = a.reshape(np.prod(a.shape))
    b = b.reshape(np.prod(b.shape))
    print "a = \n%s" % str(a)
    print "b = \n%s" % str(b)
    assert(0)

def gc():
  global W, dW, S, dS
  all_data = [None, (0, None), (4, None, None), \
              (4, (1, None), None), \
              (3, (4, None, None), (0, (3, None, None)))]
  shape = [w.shape for w in W]
  eps = 1e-7
  Worg = [copy.copy(w.reshape(np.prod(w.shape))) for w in W]
  y = 5
  for data in all_data:
    (output, scratch), pred = fp(data)
    clear_dW()
    bp(output, scratch, y, data)
    print "Verifing gradient for data:", data
    for j in xrange(len(dW)):
      print "Over symbol %d" % j
      dW_numeric = np.zeros_like(Worg[j])
      for i in xrange(Worg[j].shape[0]):
        W[j] = copy.copy(Worg[j])
        W[j][i] += eps
        W[j] = W[j].reshape(shape[j])
        (output0, scratch0), pred0 = fp(data)

        W[j] = copy.copy(Worg[j])
        W[j][i] -= eps
        W[j] = W[j].reshape(shape[j])
        (output1, scratch1), pred1 = fp(data)
        dW_numeric[i] = (-np.log(output0[y]) - -np.log(output1[y])) / (2 * eps)
      assert_close(dW_numeric, dW[j])

def clear_dW():
  for i in xrange(len(dW)):
    dW[i][:] = 0

def training(all_data):
  global W, dW
  idx = [0] * len(all_data)
  print "_" * 100
  print "Training"
  for epoch in xrange(100):
    score = 0
    for y in xrange(len(all_data)):
      data = all_data[y][idx[y]]
      (output, scratch), pred = fp(data)
      score += pred == y
      clear_dW()
      bp(output, scratch, y, data)
      update_weights()
      idx[y] = (idx[y] + 1) % len(all_data[y])
    print "epoch = %d, acc = %d / %d" % (epoch, score, len(all_data))
 
def main():
  data = init()
  if GC:
    gc()
  else:
    training(data)

if __name__ == '__main__':
  main()
