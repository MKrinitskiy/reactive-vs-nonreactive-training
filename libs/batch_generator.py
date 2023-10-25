import threading
import numpy as np
import pandas as pd
import cv2

from sklearn.utils import shuffle

from . batch import Batch
from . threadsafe_iterator import ThreadsafeIterator

import imgaug.augmenters as iaa


def get_object_index(objects_count):
    """Cyclic generator of indices from 0 to objects_count
    """
    current_id = 0
    while True:
        yield current_id
        current_id = (current_id + 1) % objects_count


class Dataset:
    def __init__(self, index_fname, batch_size=32, do_shuffle=True, caching=False, debug=False, augment = True):
        self.batch_size = batch_size
        self.do_shuffle = do_shuffle
        self.debug = debug
        self.caching = caching
        
        self.index_frame = pd.read_csv(index_fname)
        self.shuffle_data()

        self.objects_iloc_generator = ThreadsafeIterator(get_object_index(self.index_frame.shape[0]))
        self.yield_lock = threading.Lock()  # mutex for generator yielding of batch

        if augment:
            self.augmenter = iaa.Sequential([iaa.Affine(rotate=(-30, 30),
                                                        scale = (0.8,1.2),
                                                        translate_px=(-100,100),
                                                        cval=255)],
                                                        random_order=False)
        else:
            self.augmenter = iaa.Identity()

        self.batch = Batch()


    def __len__(self):
        return self.index_frame.shape[0]

    def shuffle_data(self):
        if self.do_shuffle:
            self.index_frame = shuffle(self.index_frame)

    def get_data_by_index(self, index):
        image = self.get_image(index)
        target = self.index_frame.iloc[index]['target']

        return image, target

    def get_image(self, index):
        if self.debug:
            # generate random image - its development stub
            img = np.random.randn(12, 17)
            img = (img-img.min())/(img.max()-img.min())
            img = (img*255).astype(np.uint8)
            img = cv2.resize(img, (550, 720), cv2.INTER_LANCZOS4)
            return img.astype('float')
        else:
            fname = self.index_frame.iloc[index]['fname']
            img = cv2.imread(fname)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.squeeze(img)
            return img.astype('float')


    def __iter__(self):
        while True:
            for obj_iloc in self.objects_iloc_generator:
                image, target = self.get_data_by_index(obj_iloc)
                image = self.augmenter(images=[image])[0]

                # Concurrent access by multiple threads to the lists below
                with self.yield_lock:
                    if len(self.batch) < self.batch_size:
                        self.batch.append(image, target)

                    if len(self.batch) >= self.batch_size:
                        yield self.batch
                        self.clean_batch()

    def clean_batch(self):
        self.batch = Batch()