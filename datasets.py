""" Contains utilities to load, generate datasets for use
    in model training.
"""
import cv2
import os
import pandas
import shutil

import numpy as np

from scipy.stats.mstats import mquantiles

class Dataset(object):
    """Dataset is a wrapper around an on disk dataset of the form
       dataset_1/
         images/
           1.jpg
           2.jpg
           ...
           30000.jpg
           ...
         labels.npy
         training_indexes.npy
         validation_indexes.npy
         testing_indexes.npy
 
        Note: Use the prepare_dataset method in this package to convert 
        a dataset of the form below to the form above.
        dataset_1/ 
          IMG/
            center_...
            left_...
            right_...
            ...
          driving_log.csv
    """
    def __init__(self, 
                 image_path, 
                 training_indexes,
                 testing_indexes,
                 validation_indexes,
                 labels):
        self.image_path = image_path
        self.training_indexes = training_indexes
        self.testing_indexes = testing_indexes
        self.validation_indexes = validation_indexes
        self.labels = labels

    def get_image_shape(self):
        raise NotImplementedError

    def get_training_size(self):
        return len(self.training_indexes)

    def get_testing_size(self):
        return len(self.testing_indexes)

    def get_validation_size(self):
        return len(self.validation_indexes)

    def get_training_labels(self):
        return self.labels[self.training_indexes]

    def get_testing_labels(self):
        return self.labels[self.testing_indexes]

    def get_validation_labels(self):
        return self.labels[self.validation_indexes]

    def training_generator(self, batch_size):
        return InfiniteImageLoadingGenerator(
            self.image_path, 
            self.training_indexes, 
            self.labels, 
            batch_size, 
            True,
            timesteps=None)

    def testing_generator(self, batch_size):
        return InfiniteImageLoadingGenerator(
            self.image_path, 
            self.testing_indexes, 
            self.labels, 
            batch_size, 
            False,
            timesteps=None)

    def validation_generator(self, batch_size):
        return InfiniteImageLoadingGenerator(
            self.image_path, 
            self.validation_indexes, 
            self.labels, 
            batch_size, 
            False,
            timesteps=None)

    def sequential_generator(self):
        max_index = np.max([
            self.training_indexes.max(),
            self.testing_indexes.max(),
            self.validation_indexes.max()])
        indexes = np.arange(1, max_index + 1)
        return InfiniteImageLoadingGenerator(
            self.image_path, 
            indexes, 
            self.labels, 
            batch_size, 
            False,
            timesteps=None)


    def get_baseline_mse(self):
        dummy_predictor = self.get_training_labels().mean()
        mse = ((self.get_testing_labels() - dummy_predictor) ** 2).mean()
        return mse
    
class InfiniteImageLoadingGenerator(object):
    def __init__(self, 
                 image_path, 
                 indexes, 
                 labels, 
                 batch_size, 
                 shuffle_on_exhaust,
                 timesteps=None,
                 scale=1.0,
                 pctl_sampling=False):
        self.image_path = image_path
        self.indexes = indexes
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle_on_exhaust = shuffle_on_exhaust
        self.timesteps = timesteps
        self.scale = scale
        self.current_index = 0
        self.label_shape = [1]
        self.image_shape = list(self.__load_image(image_path, indexes[0]).shape)
        self.pctl_sampling = pctl_sampling
        if self.pctl_sampling:
            these_labels = labels[indexes]
            pctl_splits = mquantiles(these_labels, np.arange(0.0, 1.01, 0.01))
            self.pctl_indexes = list(filter(len, [
                indexes[np.where((these_labels >= lb) & (these_labels < ub))[0]]
                for lb, ub in zip(pctl_splits[:-1], pctl_splits[1:])]))

    def __load_image(self, images_path, index):
        image_path = os.path.join(images_path, "%s.jpg" % index)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])

        return ((image - (255.0/2))/255.0)

    def with_timesteps(self, timesteps):
        return InfiniteImageLoadingGenerator(
            self.image_path, 
            self.indexes, 
            self.labels, 
            self.batch_size, 
            self.shuffle_on_exhaust,
            timesteps=timesteps)

    def with_scale(self, scale):
        return InfiniteImageLoadingGenerator(
            self.image_path, 
            self.indexes, 
            self.labels, 
            self.batch_size, 
            self.shuffle_on_exhaust,
            self.timesteps,
            scale=scale)

    def with_pctl_sampling(self):
        return InfiniteImageLoadingGenerator(
            self.image_path, 
            self.indexes, 
            self.labels, 
            self.batch_size, 
            self.shuffle_on_exhaust,
            self.timesteps,
            self.scale,
            pctl_sampling=True)

    def get_image_shape(self):
        return self.image_shape

    def get_size(self):
        return len(self.indexes)

    def get_batch_size(self):
        return self.batch_size

    def __iter__(self):
        return self

    def incr_index(self, n=1):
        if self.current_index + n == len(self.indexes):
            self.current_index = 0
            if self.shuffle_on_exhaust:
                np.random.shuffle(self.indexes)
        else:
            self.current_index += n 
        return self.current_index
 
    def skip(self, n):
        self.incr_index(n)
        return self

    def __next__(self):
        """ This method generates batches of data from 
            training_indexes.npy, validation_indexes.npy, testing_indexes.npy
            
            for example:
            Lets suppose
            batch_size : 1
            training_indexes : [100, 28, 32, 48, 56, 64, 73, 89...]
                                 ^ 
                                current_index
           If timesteps are required:
           This method will return (1, 10, 80, 320, 3) output, it basically
           returns a numpy array containing images from 91-100.
           else
           This method will return (1, 80, 320, 3) output, it basically
           returns a numpy array containing image number 100.

           This method advances the current index and wraps around once the 
           training_indexes array is exhausted.(training_indexes is shuffled
           on wrap around).

           NOTE: When image data is loaded it is normalized in the load_image
                 method.
        """
        labels = np.empty([self.batch_size] + self.label_shape)
        if self.timesteps:
            images = np.empty([self.batch_size, self.timesteps] + self.image_shape)
        else:
            images = np.empty([self.batch_size] + self.image_shape)

        if self.pctl_sampling:
            max_bins = max(self.batch_size, len(self.pctl_indexes))
            per_bin = max(self.batch_size / max_bins, 1)
            index_bins = np.random.choice(self.pctl_indexes, max_bins)
            next_indexes = []
            for index_bin in index_bins:
                remaining = self.batch_size - len(next_indexes)
                for_this_bin = min(remaining, per_bin)
                next_indexes.extend(np.random.choice(index_bin, for_this_bin))
        else:
            next_indexes = [
                self.indexes[self.incr_index()]
                for _ in np.arange(self.batch_size)]

        for i, next_index in enumerate(next_indexes):
            labels[i] = self.labels[next_index] * self.scale
            if self.timesteps:
                for step in np.arange(self.timesteps):
                    images[i, self.timesteps - step - 1] = self.__load_image(
                        self.image_path, int(max(0, next_index - step)))
            else:
                images[i] = self.__load_image(self.image_path, next_index)

        return (images, labels)

def prepare_dataset(
    input_dir,
    output_dir,
    training_percent,
    testing_percent,
    validation_percent):
    """
      FROM:
      dataset_1/ 
        IMG/
          center_...
          left_...
          right_...
          ...
        driving_log.csv

       TO:
       dataset_1/
         images/
           1.jpg
           2.jpg
           ...
           30000.jpg
           ...
         labels.npy
         training_indexes.npy
         validation_indexes.npy
         testing_indexes.npy
 
      Splits to train, test and validation.
    """

    input_log_csv = os.path.join(input_dir, "driving_log.csv")
    
    output_images_dir = os.path.join(output_dir, "images")
    if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)
    input_log = pandas.read_csv(input_log_csv).as_matrix()
    labels = input_log[:, 3]
    filenames = input_log[:, 0]
    for i, filename in enumerate(filenames):
        shutil.copyfile(
            os.path.join(input_dir, filename), 
            os.path.join(output_dir, "images/%s.jpg" % i))
     
    indexes = np.arange(len(labels))
    np.random.shuffle(indexes)
    training_size = int(len(indexes) * training_percent)
    testing_size = int(len(indexes) * testing_percent)
    training_indexes = indexes[0 : training_size] 
    testing_indexes = indexes[training_size : training_size + testing_size]
    validation_indexes = indexes[training_size + testing_size : len(indexes)]

    np.save(os.path.join(output_dir, "labels.npy"), labels)
    np.save(
        os.path.join(output_dir, "training_indexes.npy"), training_indexes)
    np.save(
        os.path.join(output_dir, "validation_indexes.npy"), validation_indexes)
    np.save(
        os.path.join(output_dir, "testing_indexes.npy"), testing_indexes)

def load_dataset(dataset_dir):
    return Dataset(
        os.path.join(dataset_dir, "images"),
        np.load(os.path.join(dataset_dir, "training_indexes.npy")),
        np.load(os.path.join(dataset_dir, "testing_indexes.npy")),
        np.load(os.path.join(dataset_dir, "validation_indexes.npy")),
        np.load(os.path.join(dataset_dir, "labels.npy")))

if __name__ == "__main__":
    dataset_name='dataset_32'
    prepare_dataset("../raw_data/" + dataset_name, "../datasets/" + dataset_name, 0.75, 0.15, 0.1)
    dataset = load_dataset("../datasets/" + dataset_name)
    generator = dataset.training_generator(32).with_timesteps(10).with_pctl_sampling()
    print(generator.__next__()[0].shape)
    print(generator.__next__()[1].shape)

