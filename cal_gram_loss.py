# coding=UTF-8
import vgg
import tensorflow as tf
import numpy as np
import utils
import scipy
import os
from argparse import ArgumentParser
from PIL import Image

# ---- 关键参数------ #
parser = ArgumentParser()
parser.add_argument('--imgpath_a', default='./imgs/monet1.png')  # 图片a路径
parser.add_argument('--imgpath_b', default='./imgs/monet2.png')  # 图片b路径
parser.add_argument('--network',default='imagenet-vgg-verydeep-19.mat')  # 所需的vgg19网络文件路径
parser.add_argument('--pooling', default='max')  # 池化方式
parser.add_argument('--outputs', default='./outputs')  # 输出路径
parser.add_argument('--scale', default=True)  # True：对图片缩放  False：不缩放
parser.add_argument('--width', default=512)  # 如果对图片缩放，缩放后图像宽度=width
opt = parser.parse_args()
# ------------------- #


STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1') # 用来计算gram矩阵的层


image_a = utils.imread(opt.imgpath_a) # 读取图片
image_b = utils.imread(opt.imgpath_b) # 读取图片

if opt.scale: # 放缩图片
    image_a = scipy.misc.imresize(image_a,(int(1.0*image_a.shape[0]*opt.width/image_a.shape[1]), opt.width))
    image_b = scipy.misc.imresize(image_b,(int(1.0*image_a.shape[0]*opt.width/image_a.shape[1]), opt.width))

vgg_weights, vgg_mean_pixel = vgg.load_net(opt.network) # 加载vgg19网络
style_feature_a = {} # 用来存储特征图
style_feature_b = {} # 用来存储特征图

# ---- 计算gram矩阵------ #
g = tf.Graph()
with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
    # 计算图像a的gram矩阵
    image = tf.placeholder('float', shape=(1,)+image_a.shape)
    net = vgg.net_preloaded(vgg_weights, image, opt.pooling)
    style_pre = np.array([vgg.preprocess(image_a, vgg_mean_pixel)])
    for layer in STYLE_LAYERS:
        features = net[layer].eval(feed_dict={image: style_pre})
        features = np.reshape(features, (-1, features.shape[3]))
        gram = np.matmul(features.T, features) / features.size # 计算gram矩阵的公式
        style_feature_a[layer] = gram

    # 计算图像b的gram矩阵
    image = tf.placeholder('float', shape=(1,) + image_b.shape)
    net = vgg.net_preloaded(vgg_weights, image, opt.pooling)
    style_pre = np.array([vgg.preprocess(image_b, vgg_mean_pixel)])
    for layer in STYLE_LAYERS:
        features = net[layer].eval(feed_dict={image: style_pre})
        features = np.reshape(features, (-1, features.shape[3])) # 计算gram矩阵的公式
        gram = np.matmul(features.T, features) / features.size
        style_feature_b[layer] = gram
# ------------------- #

style_losses = []
for style_layer in STYLE_LAYERS:
    gram_a = style_feature_a[style_layer]
    gram_b = style_feature_b[style_layer]
    subgram = gram_a - gram_b # 两个gram矩阵的差值
    style_losses.append(np.sum(subgram*subgram) / gram_a.size) # 平方和差异
style_loss = np.sum(style_losses)


# ---- 输出结果，写入txt文件------ #
utils.mkdir(opt.outputs)
for style_layer in STYLE_LAYERS:
    np.savetxt(os.path.join(opt.outputs, 'a_%s.txt' % style_layer), style_feature_a[style_layer]) # 图像a的各个层gram矩阵
    np.savetxt(os.path.join(opt.outputs, 'b_%s.txt' % style_layer), style_feature_b[style_layer]) # 图像b的各个层gram矩阵
np.savetxt(os.path.join(opt.outputs, 'style_loss.txt'), [style_loss])
print('style_loss = %f'%style_loss)
print('finish')
# ---- 程序结束------ #

