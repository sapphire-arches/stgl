from PIL import Image
import numpy as np


def fit_channel(channel_img):
    ys = [channel_img.getpixel((i, 0)) for i in range(r.size[0])]


def curve_fit(img):
    r = img.getchannel("R")
    g = img.getchannel("G")
    b = img.getchannel("B")
    channels = [r, g, b]

def main():
    from sys import argv
    if len(argv) < 2:
        print("I need a file name\n")
        return
    img = Image.open(argv[1])
    print(curve_fit(img))
    img.close()


if __name__ == '__main__':
    main()
