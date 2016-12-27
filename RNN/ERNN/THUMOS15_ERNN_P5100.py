__author__ = 'yuanjun'

import numpy as np
import scipy.io as sio

import theano
import theano.tensor as T

from ERNN_PKG import adadelta, E_RNN, prepare_batch

# region NumPy Test
data = sio.loadmat('P5D14_P110_PY_nodrop.mat')
dataW = sio.loadmat('../../data/p5scW0308_P14.mat')

nSeq = data['trFEAT_CAT'][0].shape[0]

X = np.ndarray((nSeq, 110, 909), dtype=theano.config.floatX)
y = np.ndarray((nSeq, 110), dtype=np.int32)
M = np.ndarray((nSeq, 110), dtype=theano.config.floatX)
for i in range(nSeq):
    X[i, :, :] = data['trFEAT_CAT'][0,i].astype(theano.config.floatX)
    y[i, :] = data['trLBL_CAT'][0,i].transpose().astype(np.int32) - 1
    M[i, :] = data['trMSK_CAT'][0,i].transpose().astype(theano.config.floatX)

H = np.tile(np.array([0.], dtype=theano.config.floatX), (X.shape[0], 21))

id = np.random.permutation(nSeq)
X = X[id,:,:]
y = y[id,:]
M = M[id,:]
wt_y = (1. / data['wt_tr'].flatten()).astype(theano.config.floatX)
wt_y[0] = wt_y[0] * 1.


nSeq1 = data['tsFEAT_CAT'][0].shape[0]

X1 = np.ndarray((nSeq1, 110, 909), dtype=theano.config.floatX)
y1 = np.ndarray((nSeq1, 110), dtype=np.int32)
M1 = np.ndarray((nSeq1, 110), dtype=theano.config.floatX)
for i in range(nSeq1):
    X1[i, :, :] = data['tsFEAT_CAT'][0,i].astype(theano.config.floatX)
    y1[i, :] = data['tsLBL_CAT'][0,i].transpose().astype(np.int32) - 1
    M1[i, :] = data['tsMSK_CAT'][0,i].transpose().astype(theano.config.floatX)

H1 = np.tile(np.array([0.], dtype=theano.config.floatX), (X1.shape[0], 21))

scW = dataW['p5scW0308_P14']

Wx = scW[0:-1, :].astype(theano.config.floatX)
bh = scW[-1].astype(theano.config.floatX)
Wh = np.zeros((21,21)).astype(theano.config.floatX)
W  = np.eye(21,21).astype(theano.config.floatX)
b  = np.zeros((21,)).astype(theano.config.floatX)

rat = np.zeros((21,))
for i in range(21):
    rat[i] = 1. * sum(y1.flatten()==i) / y1.size

print rat

# endregion



# region Build Model
print 'Building Model...'
#np.random.seed(seed=123)

tX = T.ftensor3('tX')
tH = T.fmatrix('tH')
tm = T.fmatrix('tm')
ty = T.imatrix('ty')

classifier = E_RNN(seqs=tX, h0s=tH, masks=tm, dim_x=909, dim_h=21, dim_y=21, wt_y=wt_y, Wx=Wx, Wh=Wh, W=W, bh=bh, b=b)
#classifier = E_RNN(seqs=tX, h0s=tH, masks=tm, dim_x=909, dim_h=21, dim_y=21, wt_y=wt_y)

tcost = classifier.negative_log_likelihood(ty)
tpred = classifier.pred_grosslabel()
terr = classifier.errors(ty)
tgrads = T.grad(cost=tcost, wrt=classifier.params)

f_grad_shared, f_update = adadelta(classifier.params, tgrads, tX, tH, tm, ty, tcost)
f_pred_gross = theano.function([tX, tH, tm], tpred,
                         on_unused_input='ignore')
f_pred_err = theano.function([tX, tH, tm, ty], terr,

                             on_unused_input='ignore')

f_watch = theano.function([tX, tH, tm, ty], classifier.watch_var(ty),
                          on_unused_input='ignore')

# endregion

# region Training Process
print 'Training...'

train_sz = X.shape[0]
train_batch_sz = X.shape[0]
train_batch_n = train_sz / train_batch_sz

train_cost = np.zeros((2000,))
train_err  = np.zeros((2000,))
test_cost  = np.zeros((2000,))
test_err   = np.zeros((2000,))

this_besttr = 1e4
this_bestts = 1e4
for epoch in range(2000):

    for batch_idx in range(train_batch_n):

        batch_X, batch_H, batch_m, batch_y = prepare_batch(X=X, H=H, m=M, y=y, batch_idx=batch_idx, batch_sz=train_batch_sz)

        #v = f_watch(batch_X, batch_H, batch_m, batch_y)
        #cost = f_grad_shared(batch_X, batch_H, batch_m, batch_y)
        f_update()


    train_cost[epoch] = f_grad_shared(X, H, M, y)
    train_err[epoch] = f_pred_err(X, H, M, y)
    test_cost[epoch] = f_grad_shared(X1, H1, M1, y1)
    test_err[epoch] = f_pred_err(X1, H1, M1, y1)

    output_pred = f_pred_gross(X1, H1, M1)
    output_pred[M1==0] = -1

    rat = np.zeros((21,))
    for i in range(21):
        rat[i] = 1. * sum(output_pred.flatten()==i) / output_pred.size

    print epoch
    print train_cost[epoch], train_err[epoch]
    print test_cost[epoch], test_err[epoch]
    print rat

v1 = f_watch(X, H, M, y)

# endregion
sio.savemat('P5ERNN_pram100_s.mat', {'Wx':v1[0], 'Wh':v1[1], 'W':v1[2], 'bh':v1[3], 'b':v1[4], 'pred':output_pred,
                                     'train_cost':train_cost, 'train_err':train_err, 'test_cost':test_cost, 'test_err':test_err})
