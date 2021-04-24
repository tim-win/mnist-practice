import os

from torch import Tensor
from torch.utils.data import DataLoader

MNIST_PATH = '/home/gregor/workspace/custom-datasets/mnist'


def read_binary(file_name):
    path = os.path.join(MNIST_PATH, file_name)
    with open(path, 'rb') as f:
        data = f.read()
    return data


def generate_dataset(labels_file, images_file, verbose=False):
    # http://yann.lecun.com/exdb/mnist/

    data = read_binary(images_file)
    if verbose:
        print(len(data))
        print('data[0], always zero:', data[0])
        print('data[1], always zero:', data[1])
        print('data[2], 8=unsignedb:', data[2])
        print('data[3], 3=tensors  :', data[3])

        print('data[4:8], count    :', int.from_bytes(data[4:8], 'big', signed=False))
        print('data[8:12], rows    :', int.from_bytes(data[8:12], 'big', signed=False))
        print('data[8:12], columns    :', int.from_bytes(data[12:16], 'big', signed=False))
        print('data[16:16+rows*columns]:')
    count = int.from_bytes(data[4:8], 'big', signed=False)
    rows = int.from_bytes(data[8:12], 'big', signed=False)
    columns = int.from_bytes(data[12:16], 'big', signed=False)
    _s = rows * columns
    init = 16

    images = []
    for i in range(count):
        img_data = data[init + _s*i: init+_s*(i+1)]
        images.append(Tensor([j for j in img_data]).view([28,28]))

    label_data = read_binary(labels_file)
    if verbose:
        print('first byte, always zero :', label_data[0])
        print('second byte, always zero:', label_data[1])
        print('third byte, storage type:', label_data[2])
        print('fourth, data type       :', label_data[3])

        print('fifth, count            :', int.from_bytes(label_data[4:8], 'big'))
        print('first_label:            :', label_data[8])

    offset = 8
    count = int.from_bytes(label_data[4:8], 'big')
    for x in range(count):
        images[x] = (images[x], label_data[offset+x])

    if verbose:
        print(''.join([
                     str(images[0][1].flatten()[x]).ljust(4) if x%28 else str(images[0][1].flatten()[x])+'\n\n' for x in range(len(images[0][1].flatten()))]
            ))
        print(images[0])

    return images