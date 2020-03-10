import os
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

class WGAN_GP(object):
    def __init__(self, n_rows=28, n_cols=28, chns=1, latent_dim=100):
        self.n_rows=n_rows
        self.n_cols=n_cols
        self.chns=chns
        self.latent_dim = 100
        self.image_shape = (self.n_rows, self.n_cols, self.chns)
        self.grad_weight = 10.0

        self.gen_opt = tf.keras.optimizers.Adam(lr=2e-4, beta_1=0.5,
            decay=0.0005)
        self.critic_opt = tf.keras.optimizers.Adam(lr=2e-4, beta_1=0.5,
            decay=0.0005)

        self.critic = self._build_critic()
        self.generator = self._build_generator()

    def wasserstein_loss(self, y_true, y_pred):
        return tf.reduce_mean(y_true) - tf.reduce_mean(y_pred)

    def gradient_penalty(self, x, x_hat):
        episilon = tf.random.uniform([x.shape[0], 1, 1, 1], 0.0, 1.0)
        u_hat = episilon * x + (1 - episilon) * x_hat

        with tf.GradientTape() as penalty_tape:
            penalty_tape.watch(u_hat)
            func = self.critic(u_hat)

        grads = penalty_tape.gradient(func, u_hat)
        norm_grads = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        regularizer = tf.reduce_mean((norm_grads - 1) ** 2)

        return regularizer

    @tf.function
    def train_step(self, images, grad_weight):
        noise = tf.random.normal([images.shape[0], self.latent_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as critic_tape:
            fake_images = self.generator(noise, training=True)

            # calculate critic outputs
            real_output = self.critic(images, training=True)
            fake_output = self.critic(fake_images, training=True)

            # calculate generator and critic losses
            regularizer = self.gradient_penalty(images, fake_images)
            critic_loss = self.wasserstein_loss(real_output, fake_output) + grad_weight * regularizer
            gen_loss = tf.reduce_mean(fake_output)

        # calculate gradients
        gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)

        # apply update
        self.gen_opt.apply_gradients(zip(gen_grads, self.generator.trainable_variables))
        self.critic_opt.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        return critic_loss, gen_loss, regularizer

    def train(self, input_tensor, epochs, batch_size, grad_weight):
        input_tensor = (input_tensor - 127.5) / 127.5
        input_tensor = input_tensor.astype('float32')
        input_tensor = np.expand_dims(input_tensor, axis=3)

        NUM_SAMPLES = input_tensor.shape[0]
        train_dataset = tf.data.Dataset.from_tensor_slices(input_tensor)

        # Btach and shuffle the data
        train_dataset = train_dataset.shuffle(NUM_SAMPLES).batch(
            batch_size, drop_remainder=True)

        losses = {'critic': [], 'generator': [], 'regularizer': []}

        for epoch in range(epochs):
            if (epoch % 10 == 0):
                perc = np.around(100*epoch/epochs, decimals=1)
                print('Epoch: %i. Training %1.1f%% complete.' % (epoch, perc))

            for image_batch in train_dataset:
                critic_loss, gen_loss, reg = self.train_step(image_batch, grad_weight)
                losses['critic'].append(critic_loss)
                losses['generator'].append(gen_loss)
                losses['regularizer'].append(reg)

        return self.generator, losses

    def _build_generator(self):
        DIM = 32
        AUX = int(self.n_rows / 4)

        model = tf.keras.Sequential()

        model.add(layers.Dense(units=2*AUX*AUX*DIM,
            input_shape=(self.latent_dim,)))
        model.add(layers.Activation('relu'))

        model.add(layers.Reshape(target_shape=(AUX, AUX, 2*DIM)))
        model.add(layers.Conv2DTranspose(filters=64, kernel_size=(4, 4),
            strides=(1, 1), use_bias=False, activation='relu', padding='same'))
        model.add(layers.BatchNormalization())

        model.add(layers.Conv2DTranspose(filters=32, kernel_size=(4, 4),
            strides=(2, 2), use_bias=False, activation='relu', padding='same'))
        model.add(layers.BatchNormalization())

        model.add(layers.Conv2DTranspose(filters=32, kernel_size=(4, 4),
            strides=(2, 2), use_bias=False, activation='relu', padding='same'))
        model.add(layers.BatchNormalization())

        model.add(layers.Conv2DTranspose(filters=1, kernel_size=(4, 4),
            strides=(1, 1), use_bias=False, activation='tanh', padding='same'))

        return model

    def _build_critic(self):
        DIM = 32
        model = tf.keras.Sequential()

        model.add(layers.Conv2D(filters=DIM, kernel_size=(4, 4),
            strides=(2, 2), input_shape=self.image_shape, use_bias=False,
            padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.BatchNormalization())

        model.add(layers.Conv2D(filters=2*DIM, kernel_size=(4, 4),
            strides=(2, 2), use_bias=False, padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.BatchNormalization())

        model.add(layers.Conv2D(filters=4*DIM, kernel_size=(4, 4),
            strides=(2, 2), use_bias=False, padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.BatchNormalization())

        model.add(layers.Flatten())
        model.add(layers.Dense(units=1, use_bias=False))

        return model
