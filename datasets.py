import cv2
import os
import pandas
import shutil

import numpy as np

class Dataset(object):
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
                 scale=1.0):
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
            scale=scale)

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
        labels = np.empty([self.batch_size] + self.label_shape)
        if self.timesteps:
            images = np.empty([self.batch_size, self.timesteps] + self.image_shape)
        else:
            images = np.empty([self.batch_size] + self.image_shape)

        next_indexes = [
            self.indexes[self.incr_index()]
            for _ in np.arange(self.batch_size)]

        for i, next_index in enumerate(next_indexes):
            labels[i] = self.labels[next_index] * self.scale
            if self.timesteps:
                for step in np.arange(self.timesteps):
                    images[i, self.timesteps - step - 1] = self.__load_image(
                        self.image_path, max(0, next_index - step))
            else:
                images[i] = self.__load_image(self.image_path, next_index)

        return (images, labels)

def prepare_dataset(
    input_dir,
    output_dir,
    training_percent,
    testing_percent,
    validation_percent):

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
    dataset_name='dataset_9'
    prepare_dataset("../raw_data/" + dataset_name, "../datasets/" + dataset_name, 0.75, 0.15, 0.1)
    dataset = load_dataset("../datasets/" + dataset_name)
    generator = dataset.training_generator(32)
    print(generator.__next__()[0].shape)
    print(generator.__next__()[1].shape)

