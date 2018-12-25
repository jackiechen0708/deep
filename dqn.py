import numpy as np
import tensorflow as tf
import os
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, Dense
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
import gym
import random

ENV_NAME = 'Breakout-v0'
WIDTH = 84
HEIGHT = 84
NUM_EPOCH = 12000
BATCH_SIZE = 32
LEARNING_RATE = 0.00025
MOMENTUM = 0.95
RESUME = False
TRAIN = True
MODEL_PATH = 'model/breakout'


class Agent:
    def __init__(self, actions):
        self.actions = actions
        self.eps = 1.0
        self.eps_step = (1.0 - 0.1) / 1000000
        self.t = 0

        self.reward_sum = 0
        self.q_sum = 0
        self.loss_sum = 0
        self.duration = 0
        self.episode = 0

        self.history = deque()

        self.image_placeholder, self.q_placeholder, q_network = self.build()
        dqn_weights = q_network.trainable_weights

        self.target_image_placeholder, self.target_q_placeholder, target_q_network = self.build()
        target_dqn_weights = target_q_network.trainable_weights

        self.update_target = [target_dqn_weights[i].assign(dqn_weights[i]) for i in
                              range(len(target_dqn_weights))]

        self.cur_action, self.reward, self.loss, self.train_op = self.create_train_op(dqn_weights)

        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver(dqn_weights)

        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)

        self.sess.run(tf.initialize_all_variables())

        if RESUME:
            ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            else:
                print('Load Failed')

        self.sess.run(self.update_target)

    def build(self):
        DQN = Sequential()
        DQN.add(Convolution2D(32, 3, 3, subsample=(2, 2), activation='relu',
                              input_shape=(4, WIDTH, HEIGHT)))
        DQN.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
        DQN.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
        DQN.add(Convolution2D(64, 3, 3, subsample=(2, 2), activation='relu'))
        DQN.add(Convolution2D(128, 3, 3, subsample=(1, 1), activation='relu'))
        DQN.add(Convolution2D(128, 3, 3, subsample=(1, 1), activation='relu'))
        DQN.add(Convolution2D(128, 3, 3, subsample=(2, 2), activation='relu'))
        DQN.add(Convolution2D(256, 3, 3, subsample=(1, 1), activation='relu'))
        DQN.add(Convolution2D(256, 3, 3, subsample=(1, 1), activation='relu'))
        DQN.add(Flatten())
        DQN.add(Dense(1024, activation='relu'))
        DQN.add(Dense(self.actions))

        image_placeholder = tf.placeholder(tf.float32, [None, 4, WIDTH, HEIGHT])
        q_placeholder = DQN(image_placeholder)

        return image_placeholder, q_placeholder, DQN

    def create_train_op(self, dqn_weights):
        cur_action = tf.placeholder(tf.int64, [None])
        reward = tf.placeholder(tf.float32, [None])

        q_value = tf.reduce_sum(tf.multiply(self.q_placeholder, tf.one_hot(cur_action, self.actions, 1.0, 0.0)),
                                reduction_indices=1)

        error = tf.abs(reward - q_value)
        loss1 = tf.clip_by_value(error, 0.0, 1.0)
        loss2 = error - loss1
        loss = tf.reduce_mean(0.5 * tf.square(loss1) + loss2)

        optimizer = tf.train.MomentumOptimizer(LEARNING_RATE, momentum=MOMENTUM, )
        train_op = optimizer.minimize(loss, var_list=dqn_weights)

        return cur_action, reward, loss, train_op

    def init_state(self, cur_img, last_img):
        img = np.maximum(cur_img, last_img)
        img = np.uint8(resize(rgb2gray(img), (WIDTH, HEIGHT)) * 255)
        state = [img for _ in range(4)]
        return np.stack(state, axis=0)

    def take_action(self, state):
        if self.eps >= random.random() or self.t < 20000:
            action = random.randrange(self.actions)
        else:
            action = np.argmax(self.q_placeholder.eval(feed_dict={self.image_placeholder: [np.float32(state / 255.0)]}))

        if self.eps > 0.1 and self.t >= 20000:
            self.eps -= self.eps_step

        return action

    def run(self, state, action, reward, stop, cur_img):
        next_state = np.append(state[1:, :, :], cur_img, axis=0)

        reward = np.clip(reward, -1, 1)

        self.history.append((state, action, reward, next_state, stop))
        if len(self.history) > 400000:
            self.history.popleft()

        if self.t >= 20000:
            if self.t % 4 == 0:
                self.train_network()

            if self.t % 10000 == 0:
                self.sess.run(self.update_target)

            if self.t % 300000 == 0:
                self.saver.save(self.sess, MODEL_PATH + '/' + ENV_NAME, global_step=self.t)

        self.reward_sum += reward
        self.q_sum += np.max(self.q_placeholder.eval(feed_dict={self.image_placeholder: [np.float32(state / 255.0)]}))
        self.duration += 1

        if stop:
            if self.t >= 20000:
                stats = [self.reward_sum, self.q_sum / float(self.duration),
                         self.duration, self.loss_sum / (float(self.duration) / float(4))]
                for i in range(len(stats)):
                    self.sess.run(self.update_ops[i], feed_dict={
                        self.summary_placeholders[i]: float(stats[i])
                    })

            print(self.episode + 1, self.t, self.duration, self.eps,
                  self.reward_sum, self.q_sum / float(self.duration),
                  self.loss_sum / (float(self.duration) / float(4)))

            self.reward_sum = 0
            self.q_sum = 0
            self.loss_sum = 0
            self.duration = 0
            self.episode += 1

        self.t += 1

        return next_state

    def train_network(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        stop_batch = []

        for data in random.sample(self.history, BATCH_SIZE):
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])
            stop_batch.append(data[4])

        stop_batch = np.array(stop_batch) + 0

        target_q_values_batch = self.target_q_placeholder.eval(
            feed_dict={self.target_image_placeholder: np.float32(np.array(next_state_batch) / 255.0)})
        y_batch = reward_batch + (1 - stop_batch) * 0.99 * np.max(target_q_values_batch, axis=1)

        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={
            self.image_placeholder: np.float32(np.array(state_batch) / 255.0),
            self.cur_action: action_batch,
            self.reward: y_batch
        })

        self.loss_sum += loss

    def predict(self, state):
        if random.random() <= 0.05:
            action = random.randrange(self.actions)
        else:
            action = np.argmax(self.q_placeholder.eval(feed_dict={self.image_placeholder: [np.float32(state / 255.0)]}))

        self.t += 1

        return action


def preprocess(cur_img, last_img):
    img = np.maximum(cur_img, last_img)
    img = np.uint8(resize(rgb2gray(img), (WIDTH, HEIGHT)) * 255)
    return np.reshape(img, (1, WIDTH, HEIGHT))


def main():
    env = gym.make(ENV_NAME)
    agent = Agent(actions=env.action_space.n)

    if TRAIN:  # Train mode
        for _ in range(NUM_EPOCH):
            stop = False
            cur_img = env.reset()
            for _ in range(random.randint(1, 30)):
                last_img = cur_img
                cur_img, _, _, _ = env.step(0)
            state = agent.init_state(cur_img, last_img)
            while not stop:
                last_img = cur_img
                action = agent.take_action(state)
                cur_img, reward, stop, _ = env.step(action)
                img = preprocess(cur_img, last_img)
                state = agent.run(state, action, reward, stop, img)
    else:
        for _ in range(30):
            stop = False
            cur_img = env.reset()
            for _ in range(random.randint(1, 30)):
                last_img = cur_img
                cur_img, _, _, _ = env.step(0)
            state = agent.init_state(cur_img, last_img)
            while not stop:
                last_img = cur_img
                action = agent.predict(state)
                cur_img, _, stop, _ = env.step(action)
                env.render()
                img = preprocess(cur_img, last_img)
                state = np.append(state[1:, :, :], img, axis=0)


if __name__ == '__main__':
    main()
