from torch.autograd import Variable


class thread_killer(object):
    """Boolean object for signaling a worker thread to terminate"""

    def __init__(self):
        self.to_kill = False

    def __call__(self):
        return self.to_kill

    def set_tokill(self, tokill):
        self.to_kill = tokill



def threaded_batches_feeder(tokill, batches_queue, dataset_generator):
    while tokill() == False:
        for sample in dataset_generator:
            batches_queue.put(sample, block=True)
            if tokill() == True:
                return


def threaded_cuda_batches(tokill, cuda_batches_queue, batches_queue, DEVICE):
    while tokill() == False:
        sample = batches_queue.get(block=True)
        # images, targets = sample.images, sample.targets
        # image = Variable(image.float()).to(DEVICE)
        # target = Variable(target.float()).to(DEVICE)
        sample.to_tensor()
        sample.to_cuda(DEVICE, to_variable=True)
        cuda_batches_queue.put(sample, block=True)

        if tokill() == True:
            return