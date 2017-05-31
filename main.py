import os
import tensorflow as tf
import download
from data_loader import DataLoader
from models.model import Model

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('base_url', 'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/', 'dataset url')
flags.DEFINE_string('name', 'vangogh2photo', 'decide which dataset to use')
flags.DEFINE_string('phase', 'train', 'train or test')

flags.DEFINE_boolean('download_data', False, 'whether to download, extract image data')
flags.DEFINE_string('data_dir', './datasets/', 'directory path to download and extract data')

flags.DEFINE_integer('batch_size', 1, 'input batch size')
flags.DEFINE_integer('load_size', 286, 'scale images to this size')
flags.DEFINE_integer('crop_size', 256, 'then crop to this size')
flags.DEFINE_boolean('flip', True, 'randomly flip image horizontally')

flags.DEFINE_boolean('continue_train', True, 'whether to load latest checkpoint')
flags.DEFINE_string('ckpt_dir', './checkpoints/', 'directory path to checkpoint files')
flags.DEFINE_string('result_dir', './results/', 'directory path to save test result images')
flags.DEFINE_integer('print_log_freq', 100, 'frequency of printing logs')
flags.DEFINE_integer('save_log_freq', 1000, 'frequency of saving logs')
flags.DEFINE_integer('save_ckpt_freq', 1000, 'frequency of saving checkpoint')

flags.DEFINE_integer('total_epoch', 100, 'total number of epoch to train')
flags.DEFINE_integer('channel_A', 3, 'number of image A channels')
flags.DEFINE_integer('channel_B', 3, 'number of image B channels')
flags.DEFINE_float('lambda_A', 10.0, 'weight for cycle loss (A -> B -> A)')
flags.DEFINE_float('lambda_B', 10.0, 'weight for cycle loss (B -> A -> B)')
flags.DEFINE_float('lambda_idt', 0.1, 'weight scale for identity loss, multiplied by lambda_A or lambda_B')
flags.DEFINE_float('beta1', 0.5, 'adam optimizer beta1 parameter')
flags.DEFINE_float('learning_rate', 0.0002, 'initial learning rate')
flags.DEFINE_integer('pool_size', 50, 'the size of image buffer that stores previously generated images')
flags.DEFINE_string('log_dir', './logs/', 'directory path to write summary')

flags.DEFINE_integer('min_queue_examples', 1000, 'minimum number of elements in batch queue')

available_names = [
        'ae_photos',
        'apple2orange',
        'summer2winter_yosemite',
        'horse2zebra',
        'monet2photo',
        'cezanne2photo',
        'ukiyoe2photo',
        'vangogh2photo',
        'maps',
        'cityscapes',
        'facades',
        'iphone2dslr_flower',
    ]


def main(argv):
    file_name = FLAGS.name
    if file_name in available_names:
        extract_dir = os.path.join(FLAGS.data_dir, file_name)
        if not os.path.isdir(extract_dir):
            url = os.path.join(FLAGS.base_url, '{}.zip'.format(file_name))
            downloaded_file = download.maybe_download_from_url(url, FLAGS.data_dir)

            extract_dir = FLAGS.data_dir + file_name
            download.maybe_extract(downloaded_file, extract_dir)

        if FLAGS.phase == 'train':
            train()
        elif FLAGS.phase == 'test':
            test()
        else:
            print('Unexpected phase: {}'.format(FLAGS.phase))
            print('Choose train or test')
    else:
        print('Available datasets are: {}'.format(available_names))

    return


def train():
    loader = DataLoader(FLAGS)
    trainA, trainA_count = loader.load_trainA()
    trainB, trainB_count = loader.load_trainB()

    model = Model(FLAGS)
    model.train(trainA, trainB, max([trainA_count, trainB_count]))
    model.close()


def test():
    loader = DataLoader(FLAGS)
    testA, testA_count = loader.load_testA()
    testB, testB_count = loader.load_testB()

    model = Model(FLAGS)
    model.testA(testA, testA_count)
    model.testB(testB, testB_count)
    model.close()


if __name__ == '__main__':
    tf.app.run()
