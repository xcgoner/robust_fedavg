import horovod.keras as hvd

hvd.init()
print('local_rank=%d, rank=%d, size=%d' % (hvd.local_rank(), hvd.rank(), hvd.size()))