__author__ = 'yuanjun'

import numpy as np
import scipy.io as sio

import theano
import theano.tensor as T

from LSTM_PKG import adadelta, LSTM_A, prepare_batch

# region NumPy Test
data = sio.loadmat('../THUMOS_ERNN/D14_P100_PY.mat')

nSeq = data['trFEAT_CAT'][0].shape[0]

X = np.ndarray((nSeq, 100, 909), dtype=theano.config.floatX)
y = np.ndarray((nSeq, 100), dtype=np.int32)
M = np.ndarray((nSeq, 100), dtype=theano.config.floatX)
for i in range(nSeq):
    X[i, :, :] = data['trFEAT_CAT'][0,i].astype(theano.config.floatX)
    y[i, :] = data['trLBL_CAT'][0,i].transpose().astype(np.int32) - 1
    M[i, :] = data['trMSK_CAT'][0,i].transpose().astype(theano.config.floatX)

H = np.tile(np.array([0.], dtype=theano.config.floatX), (X.shape[0], 21))
C = np.tile(np.array([0.], dtype=theano.config.floatX), (X.shape[0], 21))

id = np.random.permutation(nSeq)
X = X[id,:,:]
y = y[id,:]
M = M[id,:]
wt_y = (1. / data['wt_tr'].flatten()).astype(theano.config.floatX)
wt_y[0] = wt_y[0] * 12


nSeq1 = data['tsFEAT_CAT'][0].shape[0]

X1 = np.ndarray((nSeq1, 100, 909), dtype=theano.config.floatX)
y1 = np.ndarray((nSeq1, 100), dtype=np.int32)
M1 = np.ndarray((nSeq1, 100), dtype=theano.config.floatX)
for i in range(nSeq1):
    X1[i, :, :] = data['tsFEAT_CAT'][0,i].astype(theano.config.floatX)
    y1[i, :] = data['tsLBL_CAT'][0,i].transpose().astype(np.int32) - 1
    M1[i, :] = data['tsMSK_CAT'][0,i].transpose().astype(theano.config.floatX)

H1 = np.tile(np.array([0.], dtype=theano.config.floatX), (X1.shape[0], 21))
C1 = np.tile(np.array([0.], dtype=theano.config.floatX), (X1.shape[0], 21))


dataW = sio.loadmat('../THUMOS_ERNN/scW0906_P14.mat')
scW = dataW['scW0906_P14']

Wx = scW[0:-1, :].astype(theano.config.floatX)
bh = scW[-1].astype(theano.config.floatX)

lstm_W = 0.1 * np.random.uniform(-1.0, 1.0, (909, 21*4)).astype(theano.config.floatX)
lstm_U = 0.1 * np.random.uniform(-1.0, 1.0, (21, 21*4)).astype(theano.config.floatX)
lstm_b = np.zeros((21*4,)).astype(theano.config.floatX)
W = np.eye(21,21).astype(theano.config.floatX)
b  = np.zeros((21,)).astype(theano.config.floatX)

lstm_W[:, 63:] = Wx
lstm_U[:, 63:] = np.zeros((21,21)).astype(theano.config.floatX)
lstm_b[0:21] = (5. * np.ones((21,))).astype(theano.config.floatX)
lstm_b[21:42] = (-5. * np.ones((21,))).astype(theano.config.floatX)
lstm_b[42:63] = (5. * np.ones((21,))).astype(theano.config.floatX)
lstm_b[63:] = bh

rat = np.zeros((21,))
for i in range(21):
    rat[i] = 1. * sum(y1.flatten()==i) / y1.size

print rat

# endregion


# region Build Model
print 'Building Model...'
np.random.seed(seed=123)

tX = T.ftensor3('tX')
tH = T.fmatrix('tH')
tC = T.fmatrix('tC')
tm = T.fmatrix('tm')
ty = T.imatrix('ty')

classifier = LSTM_A(seqs=tX, h0s=tH, c0s=tC, masks=tm, dim_x=909, dim_h=21, dim_y=21, wt_y=wt_y, lstm_W=lstm_W, lstm_U=lstm_U, lstm_b=lstm_b, W=W, b=b)
#classifier = LSTM_A(seqs=tX, h0s=tH, c0s=tC, masks=tm, dim_x=909, dim_h=21, dim_y=21, wt_y=wt_y)

tcost = classifier.negative_log_likelihood(ty)
tpred = classifier.pred_grosslabel()
terr = classifier.errors(ty)
tgrads = T.grad(cost=tcost, wrt=classifier.params)

f_grad_shared, f_update = adadelta(classifier.params, tgrads, tX, tH, tC, tm, ty, tcost)
f_pred_gross = theano.function([tX, tH, tC, tm], tpred,
                         on_unused_input='ignore')
f_pred_err = theano.function([tX, tH, tC, tm, ty], terr,
                             on_unused_input='ignore')
f_watch = theano.function([tX, tH, tC, tm, ty], classifier.watch_var(ty),
                          on_unused_input='ignore')

# endregion

# region Training Process
print 'Training...'

train_sz = X.shape[0]
train_batch_sz = 37
train_batch_n = train_sz / train_batch_sz

train_cost = np.zeros((2000,))
train_err  = np.zeros((2000,))
test_cost  = np.zeros((2000,))
test_err   = np.zeros((2000,))

for epoch in range(2000):


    for batch_idx in range(train_batch_n):

        batch_X, batch_H, batch_C, batch_m, batch_y = prepare_batch(X=X, H=H, C=C, m=M, y=y, batch_idx=batch_idx, batch_sz=train_batch_sz)

        #v1 = f_watch(batch_X, batch_H, batch_C, batch_m, batch_y)

        cost = f_grad_shared(batch_X, batch_H, batch_C, batch_m, batch_y)
        f_update()


    train_cost[epoch] = f_grad_shared(X, H, C, M, y)
    train_err[epoch] = f_pred_err(X, H, C, M, y)
    test_cost[epoch] = f_grad_shared(X1, H1, C1, M1, y1)
    test_err[epoch] = f_pred_err(X1, H1, C1, M1, y1)

    output_pred = f_pred_gross(X1, H1, C1, M1)
    output_pred[M==0] = -1

    rat = np.zeros((21,))
    for i in range(21):
        rat[i] = 1. * sum(output_pred.flatten()==i) / output_pred.size

    print epoch
    print test_err[epoch]
    print rat

    v1 = f_watch(X, H, C, M, y)

# endregion
sio.savemat('LSTM_pram100.mat', {'lstm_W':v1[0], 'lstm_U':v1[1], 'lstm_b':v1[2], 'W':v1[3], 'b':v1[4], 'pred100':output_pred,
                                 'train_cost':train_cost, 'train_err':train_err, 'test_cost':test_cost, 'test_err':test_err})

