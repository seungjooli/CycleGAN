from datetime import datetime
import math
import os
import tensorflow as tf
from . import networks
from .image_pool import ImagePool


class Model:
    def __init__(self, flags):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.flags = flags
        self.saver = None

        self.global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int32)
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        self.input_A = tf.placeholder(tf.float32, [None, flags.crop_size, flags.crop_size, flags.channel_A])
        self.input_B = tf.placeholder(tf.float32, [None, flags.crop_size, flags.crop_size, flags.channel_B])

        self.netG_A = networks.Generator('netG_A', flags.channel_B)
        self.netG_B = networks.Generator('netG_B', flags.channel_A)

        self.fake_A = self.netG_B.forward(self.input_B)
        self.fake_B = self.netG_A.forward(self.input_A)
        self.cyc_A = self.netG_B.forward(self.fake_B)
        self.cyc_B = self.netG_A.forward(self.fake_A)
        self.identity_A = self.netG_A.forward(self.input_B)
        self.identity_B = self.netG_B.forward(self.input_A)

        if flags.phase == 'train':
            self.netD_A = networks.Discriminator('netD_A')
            self.netD_B = networks.Discriminator('netD_B')

            self.fake_A_pool = tf.placeholder(tf.float32, [None, flags.crop_size, flags.crop_size, flags.channel_A])
            self.fake_B_pool = tf.placeholder(tf.float32, [None, flags.crop_size, flags.crop_size, flags.channel_B])

            self.pred_real_A = self.netD_A.forward(self.input_A)
            self.pred_fake_A = self.netD_A.forward(self.fake_A)
            self.pred_fake_A_pool = self.netD_A.forward(self.fake_A_pool)

            self.pred_real_B = self.netD_B.forward(self.input_B)
            self.pred_fake_B = self.netD_B.forward(self.fake_B)
            self.pred_fake_B_pool = self.netD_B.forward(self.fake_B_pool)

            self.loss_G_A = tf.reduce_mean(tf.squared_difference(self.pred_fake_B, tf.ones_like(self.pred_fake_B)))
            self.loss_G_B = tf.reduce_mean(tf.squared_difference(self.pred_fake_A, tf.ones_like(self.pred_fake_A)))

            self.loss_cycle_A = self.flags.lambda_A * tf.reduce_mean(tf.abs(self.cyc_A - self.input_A))
            self.loss_cycle_B = self.flags.lambda_B * tf.reduce_mean(tf.abs(self.cyc_B - self.input_B))

            self.loss_identity_A = self.flags.lambda_B * self.flags.lambda_idt * \
                                   tf.reduce_mean(tf.abs(self.identity_A - self.input_B))
            self.loss_identity_B = self.flags.lambda_A * self.flags.lambda_idt * \
                                   tf.reduce_mean(tf.abs(self.identity_B - self.input_A))

            self.loss_G = self.loss_G_A + self.loss_G_B + \
                          self.loss_cycle_A + self.loss_cycle_B + \
                          self.loss_identity_A + self.loss_identity_B

            self.loss_D_A_real = tf.reduce_mean(tf.squared_difference(self.pred_real_A, tf.ones_like(self.pred_real_A)))
            self.loss_D_A_fake = tf.reduce_mean(tf.squared_difference(self.pred_fake_A_pool, tf.zeros_like(self.pred_fake_A_pool)))
            self.loss_D_A = (self.loss_D_A_real + self.loss_D_A_fake) * 0.5

            self.loss_D_B_real = tf.reduce_mean(tf.squared_difference(self.pred_real_B, tf.ones_like(self.pred_real_B)))
            self.loss_D_B_fake = tf.reduce_mean(tf.squared_difference(self.pred_fake_B_pool, tf.zeros_like(self.pred_fake_B_pool)))
            self.loss_D_B = (self.loss_D_B_real + self.loss_D_B_fake) * 0.5

            self.vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='netG')
            self.vars_D_A = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='netD_A')
            self.vars_D_B = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='netD_B')

            optimizer = tf.train.AdamOptimizer(self.flags.learning_rate, beta1=self.flags.beta1)
            self.optimizer_G = optimizer.minimize(self.loss_G, var_list=self.vars_G, name='train_G')
            self.optimizer_D_A = optimizer.minimize(self.loss_D_A, var_list=self.vars_D_A, name='train_D_A')
            self.optimizer_D_B = optimizer.minimize(self.loss_D_B, var_list=self.vars_D_B, name='train_D_B')

    def get_training_summary(self):
        summaries = [
            tf.summary.image('input_A', self.input_A),
            tf.summary.image('fake_B', self.fake_B),
            tf.summary.image('fake_B_pool', self.fake_B_pool),
            tf.summary.image('cycle_A', self.cyc_A),
            tf.summary.image('identity_A', self.identity_A),

            tf.summary.image('input_B', self.input_B),
            tf.summary.image('fake_A', self.fake_A),
            tf.summary.image('fake_A_pool', self.fake_A_pool),
            tf.summary.image('cycle_B', self.cyc_B),
            tf.summary.image('identity_B', self.identity_B),

            tf.summary.histogram('pred_A_real', self.pred_real_A),
            tf.summary.histogram('pred_A_fake', self.pred_fake_A),
            tf.summary.histogram('pred_A_fake_pool', self.pred_fake_A_pool),

            tf.summary.histogram('pred_B_real', self.pred_real_B),
            tf.summary.histogram('pred_B_fake', self.pred_fake_B),
            tf.summary.histogram('pred_B_fake_pool', self.pred_fake_B_pool),

            tf.summary.scalar('loss_G', self.loss_G),
            tf.summary.scalar('loss_G_A', self.loss_G_A),
            tf.summary.scalar('loss_G_B', self.loss_G_B),
            tf.summary.scalar('loss_cycle_A', self.loss_cycle_A),
            tf.summary.scalar('loss_cycle_B', self.loss_cycle_B),
            tf.summary.scalar('loss_identity_A', self.loss_identity_A),
            tf.summary.scalar('loss_identity_B', self.loss_identity_B),

            tf.summary.scalar('loss_D_A', self.loss_D_A),
            tf.summary.scalar('loss_D_A_real', self.loss_D_A_real),
            tf.summary.scalar('loss_D_A_fake', self.loss_D_A_fake),
            tf.summary.scalar('loss_D_B', self.loss_D_B),
            tf.summary.scalar('loss_D_B_real', self.loss_D_B_real),
            tf.summary.scalar('loss_D_B_fake', self.loss_D_B_fake),
        ]
        for var in self.vars_G:
            summaries.append(tf.summary.histogram(var.name, var))
        for var in self.vars_D_A:
            summaries.append(tf.summary.histogram(var.name, var))
        for var in self.vars_D_B:
            summaries.append(tf.summary.histogram(var.name, var))
        return tf.summary.merge(summaries)

    def train(self, image_A, image_B, dataset_size):
        training_summary = self.get_training_summary()
        writer = tf.summary.FileWriter(self.flags.log_dir, self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        if self.flags.continue_train:
            self.load()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        fake_A_pool = ImagePool(self.flags.pool_size)
        fake_B_pool = ImagePool(self.flags.pool_size)
        increment_global_step_op = tf.assign_add(self.global_step, 1)

        steps_per_epoch = math.ceil(dataset_size / self.flags.batch_size)
        global_step_value = self.sess.run(self.global_step)
        current_epoch = math.floor(global_step_value / steps_per_epoch)
        current_step = global_step_value % steps_per_epoch

        self.print_log('Training start')
        while current_epoch < self.flags.total_epoch:
            while current_step < steps_per_epoch:
                if current_epoch >= self.flags.total_epoch:
                    break

                image_A_value = self.sess.run(image_A)
                image_B_value = self.sess.run(image_B)

                feed_dict = {self.input_A: image_A_value,
                             self.input_B: image_B_value,
                             self.is_training: True}

                fake_A_value, fake_B_value = self.sess.run([self.fake_A, self.fake_B], feed_dict=feed_dict)
                fake_A_pool_value = fake_A_pool.query(fake_A_value)
                fake_B_pool_value = fake_B_pool.query(fake_B_value)

                feed_dict.update({self.fake_A_pool: fake_A_pool_value,
                                  self.fake_B_pool: fake_B_pool_value,})

                self.sess.run([self.optimizer_G, self.optimizer_D_A, self.optimizer_D_B], feed_dict=feed_dict)

                global_step_value = self.sess.run(increment_global_step_op)
                current_step += 1

                if global_step_value % self.flags.print_log_freq == 0:
                    self.print_log('Training (epoch: {}, step: {}, global_step: {})'
                                   .format(current_epoch, current_step, global_step_value))

                if global_step_value % self.flags.save_log_freq == 0:
                    training_summary_value = self.sess.run(training_summary, feed_dict=feed_dict)
                    writer.add_summary(training_summary_value, global_step_value)

                if global_step_value % self.flags.save_ckpt_freq == 0:
                    self.save()

            current_epoch += 1
            current_step = 0
            self.save()

        self.print_log('Training complete')
        coord.request_stop()
        coord.join(threads)

    def testA(self, images, image_count):
        input_ = self.input_A
        output = self.fake_B
        result_dir = os.path.join(self.flags.result_dir, 'testA/')
        self._test(images, input_, output, image_count, result_dir)

    def testB(self, images, image_count):
        input_ = self.input_B
        output = self.fake_A
        result_dir = os.path.join(self.flags.result_dir, 'testB/')
        self._test(images, input_, output, image_count, result_dir)

    def _test(self, images, input_, output, image_count, result_dir):
        os.makedirs(result_dir, exist_ok=True)

        self.sess.run(tf.global_variables_initializer())
        self.load()

        input_output_concatenated = tf.concat([input_, output], axis=2)
        input_output_concatenated = tf.cast(input_output_concatenated[0] * 255, tf.uint8)
        encode = tf.image.encode_jpeg(input_output_concatenated, format='rgb', quality=100)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        self.print_log('Test start')
        for i in range(image_count):
            image_value = self.sess.run(images)
            feed_dict = {input_: image_value}
            result = self.sess.run(encode, feed_dict=feed_dict)

            with open(os.path.join(result_dir, '{}.jpg'.format(i)), 'wb') as f:
                f.write(result)

        self.print_log('Test complete')
        coord.request_stop()
        coord.join(threads)

    @staticmethod
    def print_log(msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print('[{}] {}'.format(timestamp, msg))

    def save(self):
        os.makedirs(self.flags.ckpt_dir, exist_ok=True)
        if self.saver is None:
            self.saver = tf.train.Saver()
        self.saver.save(self.sess, self.flags.ckpt_dir, global_step=self.global_step)
        self.print_log('Saved checkpoint and summary')

    def load(self):
        ckpt = tf.train.get_checkpoint_state(self.flags.ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            if self.saver is None:
                self.saver = tf.train.Saver()
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            self.print_log('Restored variables from checkpoint')

    def close(self):
        self.sess.close()
