__author__ = 'yuanjun'

import numpy as np
import scipy.io as sio

import theano
import theano.tensor as T

def adadelta(tparams, tgrads, tX, tH, tC, tm, ty, tcost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * np.asarray(0., dtype=theano.config.floatX))
                    for p in tparams]
    running_up2 = [theano.shared(p.get_value() * np.asarray(0., dtype=theano.config.floatX))
                   for p in tparams]
    running_grads2 = [theano.shared(p.get_value() * np.asarray(0., dtype=theano.config.floatX))
                      for p in tparams]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, tgrads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, tgrads)]

    f_grad_shared = theano.function([tX, tH, tC, tm, ty], tcost, updates=zgup + rg2up,
                                    on_unused_input='ignore',
                                    name='adadelta_f_grad_shared')

    updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams, updir)]

    f_update = theano.function([], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update

class LSTM_A(object): #Combined Gate I,F,O,C

    def __init__(self, seqs, h0s, c0s, masks, dim_x, dim_h, dim_y, wt_y=None, lstm_W=None, lstm_U=None, lstm_b=None, W=None, b=None):

        self.inputs = T.transpose(seqs, (1,0,2))
        self.h0 = h0s
        self.c0 = c0s
        self.masks = masks

        # parameters of the model
        if lstm_W is None:
            lstm_W = 0.1 * np.random.uniform(-1.0, 1.0, (dim_x, dim_h*4)).astype(theano.config.floatX)
        if lstm_U is None:
            lstm_U = 0.1 * np.random.uniform(-1.0, 1.0, (dim_h, dim_h*4)).astype(theano.config.floatX)
        if lstm_b  is None:
            lstm_b  = np.zeros((dim_h*4,)).astype(theano.config.floatX)
        if W is None:
            W = 0.1 * np.random.uniform(-1.0, 1.0, (dim_h, dim_y)).astype(theano.config.floatX)
        if b  is None:
            b  = np.zeros((dim_y,), dtype=theano.config.floatX)

        self.lstm_W = theano.shared(value=lstm_W, name='lstm_W', borrow=True)
        self.lstm_U = theano.shared(value=lstm_U, name='lstm_U', borrow=True)
        self.lstm_b = theano.shared(value=lstm_b, name='lstm_b', borrow=True)
        self.W = theano.shared(value=W, name='W', borrow=True)
        self.b = theano.shared(value=b, name='b', borrow=True)

        # bundle
        self.params = [self.lstm_W, self.lstm_U, self.lstm_b, self.W, self.b]

        # initialize the cost weight of different classes
        if wt_y is None:
            self.wt_y = T.as_tensor_variable(np.ones((dim_y,), dtype=theano.config.floatX), name='wt_y')
        else:
            self.wt_y = T.as_tensor_variable(wt_y, name='wt_y')

        # slice function
        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            return _x[:, n * dim:(n + 1) * dim]

        def recurrence(x_, h_, c_):
            
            preact = x_ + T.dot(h_, self.lstm_U)

            i = T.nnet.sigmoid(_slice(preact, 0, dim_h))
            f = T.nnet.sigmoid(_slice(preact, 1, dim_h))
            o = T.nnet.sigmoid(_slice(preact, 2, dim_h))
            c = T.tanh(_slice(preact, 3, dim_h))

            c = f * c_ + i * c
            h = o * T.tanh(c)

            s = T.nnet.softmax(T.dot(h, self.W) + self.b)

            return [h, c, s]

        self.X_state = T.dot(self.inputs, self.lstm_W) + self.lstm_b
        [h, c, s], updates = theano.scan(fn=recurrence,
                                      sequences=self.X_state,
                                      outputs_info=[self.h0, self.c0, None],
                                      n_steps=self.inputs.shape[0])

        self.py_given_x = T.transpose(s, (1,0,2))
        self.y_pred = T.argmax(self.py_given_x, axis=2)

        self.watch = self.params
        self.l2sqr = (self.lstm_W ** 2).sum() + (self.lstm_U ** 2).sum() + (self.W ** 2).sum()

    def watch_var(self, y):
        return self.watch

    def pred_grosslabel(self):
        return self.y_pred

    def negative_log_likelihood(self, y):
        py_given_x_mat = T.reshape(self.py_given_x, (self.py_given_x.shape[0] * self.py_given_x.shape[1], -1))
        y_vct = T.flatten(y)
        m_vct = T.flatten(self.masks)
        ce = -T.sum((T.log(py_given_x_mat) * m_vct[:,None])[T.arange(y_vct.shape[0]), y_vct] * self.wt_y[y_vct]) / T.sum(m_vct)
        return ce + 1.0e-5 * self.l2sqr

    def errors(self, y):
        m_vct = T.flatten(self.masks)
        return T.sum(T.neq(T.flatten(self.y_pred), T.flatten(y)) * m_vct) / T.sum(m_vct)

def prepare_batch(X, H, C, m, y, batch_idx, batch_sz):

    batch_X = X[batch_idx * batch_sz : (batch_idx+1) * batch_sz]
    batch_H = H[batch_idx * batch_sz : (batch_idx+1) * batch_sz]
    batch_C = C[batch_idx * batch_sz : (batch_idx+1) * batch_sz]
    batch_m = m[batch_idx * batch_sz : (batch_idx+1) * batch_sz]
    batch_y = y[batch_idx * batch_sz : (batch_idx+1) * batch_sz]

    return batch_X, batch_H, batch_C, batch_m, batch_y
