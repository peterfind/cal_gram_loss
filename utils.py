from PIL import Image, ImageFont, ImageDraw
import scipy.misc
import numpy as np
import os

def drawdic(image, dic, hight=30):
    textlist = []
    for name, value in dic.items():
        textlist.append('%s = %s' %(name, value))
    font = ImageFont.truetype("./simsun.ttc", hight-1)
    draw = ImageDraw.Draw(image)
    for idx, l in enumerate(textlist):
        draw.text(xy=(10, idx * hight), text=l, font=font, fill='#000000')
    return image

def  imread(path):
    img = scipy.misc.imread(path).astype(np.float)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img,img,img))
    elif img.shape[2] == 4:
        # PNG with alpha channel
        img = img[:,:,:3]
    return img


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)

def printoptions(opt):
    print('#-----------options-----------#')
    for name, value in vars(opt).items():
        print('%s = %s' %(name, value))
    print('#-----------------------------#')

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)