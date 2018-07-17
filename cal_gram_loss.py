# coding=UTF-8
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import vgg
import tensorflow as tf
import numpy as np
import utils
import scipy
from argparse import ArgumentParser
from PIL import Image
import matplotlib
matplotlib.use('Agg') # 不跳出图像窗口
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# ---- 关键参数------ #
parser = ArgumentParser()
parser.add_argument('--imgpath_a', default='./imgs/monet1.png')  # 图片a路径
parser.add_argument('--imgpath_b', default='./imgs/1-the-starry-night.jpg')  # 图片b路径
parser.add_argument('--network', default='../vgg-models/imagenet-vgg-verydeep-19.mat')  # 所需的vgg19网络文件路径
parser.add_argument('--pooling', default='max')  # 池化方式
parser.add_argument('--outputs', default='./outputs3')  # 输出路径
parser.add_argument('--scale', default=True)  # True：对图片缩放  False：不缩放
parser.add_argument('--width_a', default=64)  # 如果对图片缩放，缩放后图像宽度
parser.add_argument('--width_b', default=256)  # 如果对图片缩放，缩放后图像宽度
parser.add_argument('--patch_width', default=252)  # 图片b的patch宽度
parser.add_argument('--patch_height', default=64)  # 图片b的patch高度
opt = parser.parse_args()
# ------------------- #


STYLE_LAYERS = ('relu1_1', 'relu2_1')  # 用来计算gram矩阵的层

image_a = utils.imread(opt.imgpath_a)  # 读取图片
image_b = utils.imread(opt.imgpath_b)  # 读取图片
if opt.scale:  # 放缩图片
    image_a = scipy.misc.imresize(image_a, (int(1.0 * image_a.shape[0] * opt.width_a / image_a.shape[1]), opt.width_a))
    image_b = scipy.misc.imresize(image_b, (int(1.0 * image_b.shape[0] * opt.width_b / image_b.shape[1]), opt.width_b))
loss_save = np.zeros(shape=(image_b.shape[0] - opt.patch_height + 1,
                            image_b.shape[1] - opt.patch_width + 1), dtype=np.float32)

vgg_weights, vgg_mean_pixel = vgg.load_net(opt.network)  # 加载vgg19网络
style_feature_a = {}  # 用来存储特征图
style_feature_b = {}  # 用来存储特征图
utils.mkdir(opt.outputs)
# ---- 计算gram矩阵------ #
g = tf.Graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with g.as_default(), tf.Session(config=config) as sess:
    # 计算图像a的gram矩阵
    image = tf.placeholder('float', shape=(1,) + image_a.shape)
    net = vgg.net_preloaded(vgg_weights, image, opt.pooling)
    style_pre = np.array([vgg.preprocess(image_a, vgg_mean_pixel)])
    for layer in STYLE_LAYERS:
        features = net[layer].eval(feed_dict={image: style_pre})
        features = np.reshape(features, (-1, features.shape[3]))
        gram = np.matmul(features.T, features) / features.size  # 计算gram矩阵的公式
        style_feature_a[layer] = gram

    image = tf.placeholder('float', shape=(1, opt.patch_height, opt.patch_width, 3))
    net = vgg.net_preloaded(vgg_weights, image, opt.pooling)

    for pos_0 in range(image_b.shape[0] - opt.patch_height + 1):
        print('Calculate column %d/%d' % (pos_0, image_b.shape[0] - opt.patch_height))
        for pos_1 in range(image_b.shape[1] - opt.patch_width + 1):
            patch_b = image_b[pos_0:pos_0 + opt.patch_height, pos_1:pos_1 + opt.patch_width, :]
            style_pre = np.array([vgg.preprocess(patch_b, vgg_mean_pixel)])
            for layer in STYLE_LAYERS:
                features = net[layer].eval(feed_dict={image: style_pre})
                features = np.reshape(features, (-1, features.shape[3]))  # 计算gram矩阵的公式
                gram = np.matmul(features.T, features) / features.size
                style_feature_b[layer] = gram
            style_losses = []
            for style_layer in STYLE_LAYERS:
                gram_a = style_feature_a[style_layer]
                gram_b = style_feature_b[style_layer]
                subgram = gram_a - gram_b  # 两个gram矩阵的差值
                style_losses.append(np.sum(subgram * subgram) / gram_a.size)  # 平方和差异
            loss_save[pos_0, pos_1] = np.sum(style_losses)

for style_layer in STYLE_LAYERS:
    np.savetxt(os.path.join(opt.outputs, 'a_%s.txt' % style_layer), style_feature_a[style_layer])  # 图像a的各个层gram矩阵
    np.savetxt(os.path.join(opt.outputs, 'loss_save.txt'), loss_save)

# loss灰度热力图
loss_save_8bit = (255 * loss_save / np.max(loss_save)).astype(np.uint8)
loss_img = Image.fromarray(loss_save_8bit, mode='L')
loss_img.save(os.path.join(opt.outputs, 'loss_img.png'))
loss_img_rgb = loss_img.convert('RGB')


# 拼接 原图 + 热力图
image_b_rgb = Image.fromarray(image_b)
combine_img = Image.new('RGB', (image_b.shape[1] * 2, image_b.shape[0]), (0, 0, 0))
combine_img.paste(image_b_rgb, (0, 0))
combine_img.paste(loss_img_rgb, (image_b.shape[1] + opt.patch_width / 2, opt.patch_height / 2))
combine_img.save(os.path.join(opt.outputs, 'combine_img.png'))

# loss 曲线
# loss_save = np.load(os.path.join(opt.outputs, 'loss_save.txt'))  # 备用方法 防止程序异常中断
plt.plot(loss_save.reshape((-1)))  # 第一行、第二行 ***
plt.savefig(os.path.join(opt.outputs, 'loss_curve.png'))

# 3D图像
fig = plt.figure()
ax = Axes3D(fig)
x = range(loss_save.shape[1])
y = range(loss_save.shape[0])
X, Y = np.meshgrid(x, y)
Z = loss_save
ax.plot_surface(X, Y, Z, cmap=plt.cm.winter)
plt.savefig(os.path.join(opt.outputs, 'loss_3D.png'))

print('finish')
# ---- 程序结束------ #
