import numpy as np


def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = ((np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0)
        image_numpy = image_numpy * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0

    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:
        image_numpy = image_numpy[:, :, 0]

    return image_numpy
