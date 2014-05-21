#!/usr/bin/python
import os
import theano
from theano import tensor as T
import numpy as np
import copy

W = []
dW = []
S = None
dS = None
lr = 0.1

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
  return np.max(x[0], 0), x

def dnonlin((dX_in, x)):
  assert(dX_in.shape == x[0].shape)
  return dX_in * (x[0] > 0), x[1]

def tnn_fp(x):
  global W
  if x is None:
    return (W[-1],)
  elif len(x) == 2:
    root, expr0 = x
    f0 = tnn_fp(expr0)
    return nonlin((W[root] * f0[0], f0))
  elif len(x) == 3:
    root, expr0, expr1 = x
    f0 = tnn_fp(expr0)
    f1 = tnn_fp(expr1)
    f = np.tensordot(W[root], f0[0], axes=(2, 0)) * f1[0]
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
    o = np.outer(x[1][0], x[2][0])
    o = o.reshape(list(o.shape) + [1])
    dW[data[0]] += np.tensordot(o, dX_r, axes=(2, 1))
    dX0 = np.dot(np.tensordot(W[data[0]], x[1][0], axes=(2, 0)).transpose(), dX_in)
    dX1 = np.dot(np.tensordot(W[data[0]], x[2][0], axes=(1, 0)).transpose(), dX_in)
    tnn_bp((dX0, x[1]), data[1])
    tnn_bp((dX1, x[2]), data[2])

def lin_fp(x):
  global S
  return np.dot(S, x[0]), x

def lin_bp((dX_in, x)):
  global dS
  global S
  dX = np.dot(S.transpose(), dX_in)
  dS = np.outer(dX_in, x[1][0])
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
  global S, W, dW, lr
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
  hidden = 50
  ar = sorted(arity(data))
  for i in xrange(len(ar)):
    s = [hidden] * ar[i][1]
    print "Size of %d is %s" % (i, str(s))
    W.append(np.random.randn(*s) * 0.001)
    dW.append(np.zeros(tuple(s)))
  W.append(np.ones(hidden))
  S = np.random.randn(classes, hidden)
  return data

def fp(d):
  return softmax_fp(lin_fp(tnn_fp(d)))

def bp(output, scratch, y, data):
  tnn_bp(lin_bp(softmax_bp((output, scratch), y)), data)

def update_weights():
  for i in xrange(len(dW)):
    W[i] = W[i] - lr * dW[i] 
  S = S - lr * dS

def norm(x):
  return np.sqrt(np.sum(np.square(x)))

def assert_close(a, b):
  assert(norm(a - b.reshape(a.shape)) / norm(b) < 1e-4)

def gc():
  global W, dW, S, dS
  all_data = [None, (0, None), (1, None), (2, None)]
  shape = S.shape
  eps = 1e-2
  Sorg = copy.copy(S.reshape(np.prod(S.shape)))
  y = 5
  for data in all_data:
    print "Verifiing gradient for data:", data
    dS_numeric = np.zeros_like(Sorg)
    for i in xrange(Sorg.shape[0]):
      S = copy.copy(Sorg)
      S[i] += eps
      S = S.reshape(shape)
      (output0, scratch0), pred0 = fp(data)

      S = copy.copy(Sorg)
      S[i] -= eps
      S = S.reshape(shape)
      (output1, scratch1), pred1 = fp(data)
      dS_numeric[i] = (-np.log(output0[y]) - -np.log(output1[y])) / (2 * eps)

    (output, scratch), pred = fp(data)
    bp(output, scratch, y, data)
    assert_close(dS_numeric, dS)

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
  gc()
  #training(data)

if __name__ == '__main__':
  main()
