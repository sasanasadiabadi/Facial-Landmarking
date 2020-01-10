import os
import numpy as np
from random import shuffle
import scipy.misc
import json
import data_process
import random
import cv2


""" adopted from https://github.com/yuanyuanli85/Stacked_Hourglass_Network_Keras.git """

class Dataset():
    def __init__(self, jsonfile, imgpath, inres, outres, is_train):
        self.jsonfile = jsonfile
        self.imgpath = imgpath
        self.inres = inres
        self.outres = outres
        self.is_train = is_train
        self.nparts = 29
        self.anno = self._load_image_annotation()

    def _load_image_annotation(self):
        # load train or val annotation
        with open(self.jsonfile) as anno_file:
            anno = json.load(anno_file)

        val_anno, train_anno = [], []
        for idx, val in enumerate(anno):
            if val['isValidation'] == True:
                val_anno.append(anno[idx])
            else:
                train_anno.append(anno[idx])

        if self.is_train:
            return train_anno
        else:
            return val_anno

    def get_dataset_size(self):
        return len(self.anno)

    def get_color_mean(self):
        mean = np.array([0.4404, 0.4440, 0.4327], dtype=np.float)
        return mean

    def get_annotations(self):
        return self.anno

    def generator(self, batch_size, sigma=1, is_shuffle=False, epoch_end=False):
        '''
        Input:  batch_size * inres  * Channel (3)
        Output: batch_size * oures  * nparts
        '''
        train_input = np.zeros(shape=(batch_size, 3, self.inres[0], self.inres[1]), dtype=np.float)
        gt_heatmap = np.zeros(shape=(batch_size, self.nparts, self.outres[0], self.outres[1]), dtype=np.float)
        
        if not self.is_train:
            assert (is_shuffle == False), 'shuffle must be off in val model'
            
        while not epoch_end:
            if is_shuffle:
                shuffle(self.anno)

            for i, kpanno in enumerate(self.anno):
                if i == len(self.anno) - 1:
                    epoch_end = True

                _imageaug, _gthtmap = self.process_image(i, kpanno, sigma)
                _index = i % batch_size

                train_input[_index, :, :, :] = _imageaug
                gt_heatmap[_index, :, :, :] = _gthtmap 

                if i % batch_size == (batch_size - 1):
                    yield train_input, gt_heatmap

    def process_image(self, sample_index, kpanno, sigma):
        imagefile = kpanno['img_path']
        image = scipy.misc.imread(os.path.join(self.imgpath, imagefile))

        if image.ndim < 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # get center
        bb = np.array(kpanno['face'])
        kp = np.array(kpanno['landmarks'])


        cropimg = data_process.crop(image, bb, (224, 224))
        #cropimg = data_process.normalize(cropimg, self.get_color_mean())

        # transform keypoints
        transformed_kp = data_process.transform_kp(image, bb, kp, (224, 224))
        gtmap = data_process.generate_hm((224, 224), transformed_kp, 3)

        return np.swapaxes(cropimg, 0, 2), np.swapaxes(gtmap, 0, 2)

