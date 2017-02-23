from PIL import Image
import os


def read_bb_info(file_path):
    file_path += 'bb_info.txt'
    d = {}
    with open(file_path) as f:
        for line in f:
            a = line.split()
            if a[0] != 'img':
                d[a[0]] = {'x1': int(a[1]), 'y1': int(a[2]), 'x2': int(a[3]), 'y2': int(a[4])}
    return d


def cut_image(image_path):
    image_name = os.path.basename(image_path)
    folder_path = image_path.replace(image_name, '')
    im = Image.open(image_path)
    img_dict = get_image_dict(folder_path, image_name.replace('.jpg', ''))
    region = im.crop((img_dict['x1'], img_dict['y1'], img_dict['x2'], img_dict['y2']))
    return region


def get_image_dict(folder_path, image_name):
    imgs_dict = read_bb_info(folder_path)
    return imgs_dict[image_name]


def show_cropped_img(image_path):
    im = cut_image(image_path)
    Image._show(im)

# example.to.use
# show_cropped_img('/media/aymen/DATA/datasets/UECFOOD100/1/1.jpg')
