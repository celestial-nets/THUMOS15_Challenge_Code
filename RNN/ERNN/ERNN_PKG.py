import numpy as np
import scipy.io as sio

import theano
import theano.tensor as T

def adadelta(tparams, tgrads, tX, tH, tm, ty, tcost):
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

    f_grad_shared = theano.function([tX, tH, tm, ty], tcost, updates=zgup + rg2up,
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

class E_RNN(object):

    def __init__(self, seqs, h0s, masks, dim_x, dim_h, dim_y, wt_y=None, Wx=None, Wh=None, W=None, bh=None, b=None):

        self.inputs = T.transpose(seqs, (1,0,2))
        self.h0 = h0s
        self.masks = masks

        # parameters of the model
        if Wx is None:
            Wx = 0.1 * np.random.uniform(-1.0, 1.0, (dim_x, dim_h)).astype(theano.config.floatX)
        if Wh is None:
            Wh = 0.1 * np.random.uniform(-1.0, 1.0, (dim_h, dim_h)).astype(theano.config.floatX)
        if W  is None:
            W  = 0.1 * np.random.uniform(-1.0, 1.0, (dim_h, dim_y)).astype(theano.config.floatX)
        if bh is None:
            bh = np.zeros((dim_h,), dtype=theano.config.floatX)
        if b  is None:
            b  = np.zeros((dim_y,), dtype=theano.config.floatX)

        self.Wx = theano.shared(value=Wx, name='Wx', borrow=True)
        self.Wh = theano.shared(value=Wh, name='Wh', borrow=True)
        self.W  = theano.shared(value=W,  name='W',  borrow=True)
        self.bh = theano.shared(value=bh, name='bh', borrow=True)
        self.b  = theano.shared(value=b,  name='b',  borrow=True)

        # bundle
        self.params = [self.Wx, self.Wh, self.W, self.bh, self.b]

        # initialize the cost weight of different classes
        if wt_y is None:
            self.wt_y = T.as_tensor_variable(np.ones((dim_y,), dtype=theano.config.floatX), name='wt_y')
        else:
            self.wt_y = T.as_tensor_variable(wt_y, name='wt_y')

        A = np.array([[1,-1,0,0,0,0,0,0,0],\
             [-1,2,-1,0,0,0,0,0,0],\
             [0,-1,2,-1,0,0,0,0,0],\
             [0,0,-1,2,-1,0,0,0,0],\
             [0,0,0,-1,2,-1,0,0,0],\
             [0,0,0,0,-1,2,-1,0,0],\
             [0,0,0,0,0,-1,2,-1,0],\
             [0,0,0,0,0,0,-1,2,-1],\
             [0,0,0,0,0,0,0,-1,1]], dtype=theano.config.floatX)
        self.tA = T.as_tensor_variable(A)


        X_state = T.dot(self.inputs, self.Wx) + self.bh
        def recurrence(X_state_t, h_tm1):
            h_t = T.tanh(X_state_t + T.dot(h_tm1, self.Wh))
            s_t = T.nnet.softmax(T.dot(h_t, self.W) + self.b)
            return [h_t, s_t]

        [h, s], updates = theano.scan(fn=recurrence,
                                      sequences=X_state,
                                      outputs_info=[self.h0, None],
                                      n_steps=self.inputs.shape[0])

        self.watch = self.params
        self.py_given_x = T.transpose(s, (1,0,2))
        self.y_pred = T.argmax(self.py_given_x, axis=2)
        self.l2sqr = (self.Wx ** 2).sum() + (self.Wh ** 2).sum() + (self.W ** 2).sum()
        self.sm_cost = self.smth_cost()

    def smth_cost(self):
        Wr = T.reshape(T.transpose(self.Wx), [2121, 9])
        sm_cost = (T.dot(Wr, self.tA) * Wr).sum()
        return sm_cost

    def watch_var(self, y):
        return self.params

    def pred_grosslabel(self):
        return self.y_pred

    def negative_log_likelihood(self, y):
        py_given_x_mat = T.reshape(self.py_given_x, (self.py_given_x.shape[0] * self.py_given_x.shape[1], -1))
        y_vct = T.flatten(y)
        m_vct = T.flatten(self.masks)
        ce = -T.sum((T.log(py_given_x_mat) * m_vct[:,None])[T.arange(y_vct.shape[0]), y_vct] * self.wt_y[y_vct]) / T.sum(m_vct)
        return ce + 0.00001 * self.l2sqr + 0.00001 * self.sm_cost

    def errors(self, y):
        m_vct = T.flatten(self.masks)
        return T.sum(T.neq(T.flatten(self.y_pred), T.flatten(y)) * m_vct) / T.sum(m_vct)

def prepare_batch(X, H, m, y, batch_idx, batch_sz):

    batch_X = X[batch_idx * batch_sz : (batch_idx+1) * batch_sz]
    batch_H = H[batch_idx * batch_sz : (batch_idx+1) * batch_sz]
    batch_m = m[batch_idx * batch_sz : (batch_idx+1) * batch_sz]
    batch_y = y[batch_idx * batch_sz : (batch_idx+1) * batch_sz]

    return batch_X, batch_H, batch_m, batch_y