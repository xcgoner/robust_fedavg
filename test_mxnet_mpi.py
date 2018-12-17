from mpi4py import MPI


import mxnet as mx
from mxnet import nd

import numpy as np

from scipy import stats

import random

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

print('rank: %d' % (rank))

# mx.random.seed(1)
# y = nd.random_normal(0, 1, shape=(3, 4))

# sendbuf = nd.zeros((3,4)) + rank
# recvbuf = comm.gather(sendbuf, root=0)
# if rank == 0:
#     for i in range(size):
#         print(recvbuf[i])


# sendbuf = nd.zeros((3,4)) + rank*rank
# nd.waitall()
# recvbuf = comm.gather(sendbuf, root=0)
# if rank == 0:
#     agg = np.stack([x.asnumpy() for x in recvbuf], axis=0)
#     agg = np.mean(agg, axis=0)
#     print(agg)

sendbuf = nd.zeros((3,4)) + rank*rank
sendbuf_np = sendbuf.asnumpy()
recvbuf = comm.gather(sendbuf_np, root=0)
if rank == 0:
    agg = np.stack([x for x in recvbuf], axis=0)
    agg = np.mean(agg, axis=0)
    print(agg)
    print(agg.shape)

# sendbuf = nd.zeros((3,4)) + rank*rank
# sendbuf_np = sendbuf.asnumpy()
# recvbuf = comm.gather(sendbuf_np, root=0)
# if rank == 0:
#     agg = np.stack([x for x in recvbuf], axis=0)
#     agg = stats.trim_mean(agg, 2.0/size, axis=0)
#     print(agg)

# sendbuf = None
# if rank == 0:
#     sendbuf = [random.randint(1,10) for i in range(size)]
#     print(sendbuf)
# recvbuf = comm.scatter(sendbuf, root=0)
# print(recvbuf)