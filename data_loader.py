import imghdr
import os
import tensorflow as tf

random_seed = 0


class DataLoader:
    def __init__(self, flags):
        self.flags = flags
        paths = DataLoader.get_image_paths(self.flags.data_dir)
        self.paths_trainA = [path for path in paths if 'trainA' in path]
        self.paths_trainB = [path for path in paths if 'trainB' in path]
        self.paths_testA = [path for path in paths if 'testA' in path]
        self.paths_testB = [path for path in paths if 'testB' in path]

    @staticmethod
    def is_image_valid(filepath):
        return imghdr.what(filepath) is not None

    @staticmethod
    def get_image_paths(image_dir):
        image_paths = []
        for root, directories, filenames in os.walk(image_dir):
            image_paths += [os.path.join(root, filename) for filename in filenames]
        image_paths = [filepath for filepath in image_paths if DataLoader.is_image_valid(filepath)]

        return image_paths

    def load(self, paths, name, shuffle):
        def read_image(image_paths, name, shuffle):
            filename_queue = tf.train.string_input_producer(image_paths, name=name, shuffle=shuffle)
            reader = tf.WholeFileReader()
            key, value = reader.read(filename_queue)
            image = tf.image.decode_image(value)
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image.set_shape([None, None, 3])

            return image

        def preprocess(image, flags):
            preprocessed = image
            if flags.phase == 'train':
                preprocessed = tf.image.resize_images(preprocessed, [flags.load_size, flags.load_size])
                preprocessed = tf.image.resize_image_with_crop_or_pad(preprocessed, flags.crop_size, flags.crop_size)
                if flags.flip:
                    preprocessed = tf.image.random_flip_left_right(preprocessed, seed=random_seed)
            else:
                preprocessed = tf.image.resize_images(preprocessed, [flags.crop_size, flags.crop_size])

            return preprocessed

        data = preprocess(read_image(paths, name + '_input_producer', shuffle), self.flags)
        batch = tf.train.batch([data], batch_size=self.flags.batch_size, name=name + '_batch', )
        return batch

    def load_trainA(self):
        return self.load(self.paths_trainA, 'trainA', True), len(self.paths_trainA)

    def load_trainB(self):
        return self.load(self.paths_trainB, 'trainB', True), len(self.paths_trainB)

    def load_testA(self):
        return self.load(self.paths_testA, 'testA', False), len(self.paths_testA)

    def load_testB(self):
        return self.load(self.paths_testB, 'testB', False), len(self.paths_testB)
