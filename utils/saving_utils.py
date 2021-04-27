import os
import copy
import cv2
import numpy as np
from collections import OrderedDict

import torch
from utils.utils import tensor2im


def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print("----No checkpoints at given path----")
        return
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu")))
    print("----checkpoints loaded from path: {}----".format(checkpoint_path))
    return model


def load_checkpoint_mgpu(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print("----No checkpoints at given path----")
        return
    model_state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    print("----checkpoints loaded from path: {}----".format(checkpoint_path))
    return model


def save_checkpoint(model, save_path):
    print(save_path)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(model.state_dict(), save_path)


def save_checkpoints(opt, itr, net):
    save_checkpoint(
        net,
        os.path.join(opt.save_dir, "checkpoints", "itr_{:08d}_net_G.pth".format(itr)),
    )


def save_imgs(opt, itr, input_image, label_image, fake_label):
    for i in range(input_image.shape[0]):
        img_i = tensor2im(input_image[i].detach())
        label_i = tensor2im(label_image[i].detach())
        fake_label_i = tensor2im(fake_label[i].detach())
        output_grid = np.concatenate([img_i, label_i, fake_label_i], axis=1)
        output_grid = cv2.cvtColor(output_grid, cv2.COLOR_BGR2RGB)
        cv2.imwrite(
            os.path.join(
                opt.save_dir, "images", "itr_{:08d}_{:02d}.png".format(itr, i)
            ),
            output_grid,
        )
