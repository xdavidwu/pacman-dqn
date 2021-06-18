import random
from game import Agent, Directions
import graphicsUtils
from PIL import Image
from io import BytesIO
import re
import numpy as np
import tensorflow.compat.v1 as tf
import math
import time

# TODO: tf v2
tf.disable_v2_behavior()

def getFrame():
        ps = graphicsUtils._canvas.postscript(
                pagewidth=int(graphicsUtils._canvas['width']) / 2,
                width=int(graphicsUtils._canvas['width']) - 30,
                height=int(graphicsUtils._canvas['height']) - 30 - 35,
                x=15, y=15)
        return Image.open(BytesIO(ps.encode('ascii')), formats=['EPS'])

def indexToAction(i):
    return {
            0: Directions.NORTH,
            1: Directions.SOUTH,
            2: Directions.EAST,
            3: Directions.WEST,
            4: Directions.STOP
    }[i]

def legalMask(i, state):
    return Directions.STOP if i not in state.getLegalPacmanActions() else i

class RandomSaveFrameAgent(Agent):
    def __init__(self):
        self.frame = 0

    def getAction(self, state):
        action = random.choice(state.getLegalPacmanActions())
        t = time.monotonic_ns();
        screen = getFrame()
        screen.load()
        screen.save('pacman_%d.bmp' % self.frame)
        print(np.shape(np.asarray(screen)))
        #exit()
        #print(state.data._agentMoved, state.data.score)
        #print(state.generatePacmanSuccessor(action).data.score)
        print(time.monotonic_ns() - t)
        self.frame += 1
        return action #random.choice(state.getLegalPacmanActions())

class ReplayMemory():
    def __init__(self, size, state_shape):
        self.size = size
        self.idx = 0
        self.is_full = False
        self.s = np.empty([size] + state_shape, dtype=np.uint8)
        self.a = np.empty([size], dtype=np.uint8)
        self.r = np.empty([size], dtype=np.double)
        self.sp = np.empty([size] + state_shape, dtype=np.uint8)
        self.terminate = np.empty([size], dtype=np.bool8)

    def push(self, s, a, r, sp, terminate):
        self.s[self.idx] = s
        self.a[self.idx] = a
        self.r[self.idx] = r
        self.sp[self.idx] = sp
        self.terminate[self.idx] = terminate
        self.idx = self.idx + 1
        if self.idx == self.size:
            self.is_full = True
            self.idx = 0

    def batch(self, size):
        indices = np.random.choice(np.arange(self.size), size)
        return (np.take(self.s, indices, 0), np.take(self.a, indices),
                np.take(self.r, indices), np.take(self.sp, indices, 0),
                np.take(self.terminate, indices))

def initializeWeight(shape):
    length = len(shape)
    l = 1
    for i in range(length - 1):
        l *= shape[i]
    return tf.truncated_normal(shape, stddev=math.sqrt(2.0 / (l + shape[-1])))

def convolutionLayer(i, shape, strides=[1, 1, 1, 1], padding='SAME'):
    w = tf.Variable(initializeWeight(shape))
    b = tf.Variable(tf.zeros([shape[-1]]))
    return tf.nn.leaky_relu(tf.nn.conv2d(i, w, strides=strides, padding=padding) + b)

def layer(i, shape):
    w = tf.Variable(initializeWeight(shape))
    b = tf.Variable(tf.zeros([shape[-1]]))
    return tf.matmul(i, w) + b

def fullyConnectedLayer(i, shape):
    return tf.nn.leaky_relu(layer(i, shape))

class GraphicsDQNAgent(Agent):
    def config(self, layout_input):
        if layout_input == 'mediumGrid':
            discount = 0.9
            state_shape = [120, 136, 3]
            mem_size = 256
            batch_size = 64
            learning_rate = 0.0005
            epsilon = lambda episode, epoch: max(1 - epoch * 0.001, 0.05)
            x = tf.placeholder(tf.float32, [None] + state_shape)
            conv1 = convolutionLayer(x, [15, 15, 3, 4], strides=[1, 15, 15, 1], padding='VALID')
            conv_out = tf.reshape(conv1, [-1, 8 * 9 * 4])
            fc1 = fullyConnectedLayer(conv_out, [8 * 9 * 4, 64])
            y = layer(fc1, [64, 4])
            return (x, y, discount, state_shape, mem_size, batch_size,
                    learning_rate, epsilon)
        raise Exception('Layout %s not implemeted.' % layout_input)

    def __init__(self, layout_input):
        (x, y, dis, state_shape, msize, bsize, lr, eps) = \
                self.config(layout_input)
        self.episode = 0
        self.epoch = 0
        self.discount = dis
        self.epsilon = eps
        self.memory = ReplayMemory(msize, state_shape)
        self.batch_size = bsize
        self.x = x
        self.y = y
        self.yp = tf.placeholder(tf.float32, [None, 4])
        self.max_y = tf.reduce_max(self.y, 1)
        self.argmax_y = tf.argmax(y, 1)
        self.loss = tf.losses.mean_squared_error(self.yp, self.y)
        self.step = tf.train.AdamOptimizer(lr).minimize(self.loss)
        self.session = tf.InteractiveSession()
        self.previous_state = None
        self.previous_score = None
        self.previous_action = None
        tf.global_variables_initializer().run()

    def train(self):
        if not self.memory.is_full: return
        (s, a, r, sp, terminate) = self.memory.batch(self.batch_size)
        q = self.session.run(self.y, feed_dict={self.x: s})
        max_qp = self.session.run(self.max_y, feed_dict={self.x: sp})

        for i in range(self.batch_size):
            if terminate[i]:
                q[i][a[i]] = r[i]
            else:
                q[i][a[i]] = r[i] + self.discount * max_qp[i]

        [_, loss] = self.session.run([self.step, self.loss], feed_dict={self.x: s, self.yp: q})
        self.epoch += 1
        if self.epoch % 10 == 0:
            print('loss: %lf' % loss)
            print('sample: %lf %lf %lf' % (q[0][a[0]], r[0], max_qp[0]))

    def getAction(self, state):
        frame = getFrame()
        s = np.asarray(frame)
        if self.previous_state is not None:
            self.memory.push(self.previous_state, self.previous_action,
                    state.data.score - self.previous_score, s, False)
            self.train()
        a = None
        if random.random() < self.epsilon(self.episode, self.epoch):
            a = random.choice([0, 1, 2, 3])
        else:
            a = self.session.run(self.argmax_y, feed_dict={self.x: [s]})[0]
        self.previous_state = s
        self.previous_action = a
        self.previous_score = state.data.score
        return legalMask(indexToAction(a), state)

    def final(self, state):
        # sp does not matter
        adj_score = state.data.score
        if state.isLose():
            adj_score += 500
            adj_score -= 64
        self.memory.push(self.previous_state, self.previous_action,
                adj_score - self.previous_score, self.previous_state,
                True)
        self.train()
        print('epoch: %d' % self.epoch)
        print('episode: %d' % self.episode)
        self.episode += 1
        self.previous_state = None
        self.previous_score = None
        self.previous_action = None
