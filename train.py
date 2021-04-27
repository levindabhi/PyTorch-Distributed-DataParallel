import os
import sys
import time
import yaml
import cv2
import pprint
import traceback
import numpy as np

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

from data.custom_dataset_data_loader import CustomDatasetDataLoader, sample_data
from data.custom_dataset_data_loader import ValidDatasetDataLoader


from options.base_options import parser
from utils.tensorboard_utils import board_add_images
from utils.saving_utils import save_checkpoints, save_imgs
from utils.saving_utils import load_checkpoint, load_checkpoint_mgpu
from utils.distributed import get_world_size, set_seed, synchronize, cleanup

from networks import CustomNetwork
from perceptual_loss import VGGLoss


def options_printing_saving(opt):
    os.makedirs(opt.logs_dir, exist_ok=True)
    os.makedirs(opt.save_dir, exist_ok=True)
    os.makedirs(os.path.join(opt.save_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(opt.save_dir, "checkpoints"), exist_ok=True)

    # Saving options in yml file
    option_dict = vars(opt)
    with open(os.path.join(opt.save_dir, "training_options.yml"), "w") as outfile:
        yaml.dump(option_dict, outfile)

    for key, value in option_dict.items():
        print(key, value)


def training_loop(opt):

    if opt.distributed:
        local_rank = int(os.environ.get("LOCAL_RANK"))
        # Unique only on individual node.
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda:0")
        local_rank = 0

    net = CustomNetwork()
    if opt.continue_train:
        net = load_checkpoint(net, opt.net_checkpoint)
    net = net.to(device)
    net.train()

    if local_rank == 0:
        with open(os.path.join(opt.save_dir, "networks.txt"), "w") as outfile:
            print("<----Net---->", file=outfile)
            print(net, file=outfile)

    if opt.distributed:
        net = nn.parallel.DistributedDataParallel(
            net,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
        print("Going super fast with DistributedDataParallel")

    # initialize optimizer
    optimizer = optim.Adam(
        net.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0
    )

    custom_dataloader = CustomDatasetDataLoader()
    custom_dataloader.initialize(opt)
    loader = custom_dataloader.get_loader()

    if opt.do_valid:
        valid_dataloader = ValidDatasetDataLoader()
        valid_dataloader.initialize(opt)
        valid_loader = custom_dataloader.get_loader()

    if local_rank == 0:
        dataset_size = len(custom_dataloader)
        print("Total number of data point avaliable for training: %d" % dataset_size)
        writer = SummaryWriter(opt.logs_dir)
        print("Entering training loop!")

    # loss function
    loss_CE = nn.BCEWithLogitsLoss().to(device)
    loss_VGG = VGGLoss().to(device)

    pbar = range(opt.iter)
    get_data = sample_data(loader)

    if opt.do_valid:
        get_valid_data = sample_data(valid_loader)

    start_time = time.time()
    # Main training loop
    for itr in pbar:
        data_batch = next(get_data)
        input_tensor, label_tensor = data_batch
        input_tensor = Variable(input_tensor.to(device))
        label_tensor = Variable(label_tensor.to(device))

        output_tensor = net(input_tensor)

        loss = loss_CE(output_tensor, label_tensor)
        total_loss = loss  # weighted sum if multiple loss

        # optimized ops of zero_grad()
        for param in net.parameters():
            param.grad = None

        total_loss.backward()
        if opt.clip_grad != 0:
            nn.utils.clip_grad_norm_(net.parameters(), opt.clip_grad)
        optimizer.step()

        if opt.do_valid and itr % opt.valid_freq == 0:
            total_valid_loss = 0
            for val_i in range(opt.valid_steps):
                data_batch = next(get_valid_data)
                input_tensor, label_tensor = data_batch
                input_tensor = Variable(input_tensor.to(device))
                label_tensor = Variable(label_tensor.to(device))
                with torch.no_grad():
                    net_out = net(input_tensor)
                total_valid_loss += loss_CE(net_out, label_tensor).item()

            if local_rank == 0:
                pprint.pprint("[Valid loss-{:.6f}]".format(total_valid_loss))
                writer.add_scalar("valid_loss", total_valid_loss, itr)

        if local_rank == 0:
            # printing and saving work
            if itr % opt.print_freq == 0:
                pprint.pprint(
                    "[step-{:08d}] [time-{:.3f}] [total_loss-{:.6f}]  [loss0-{:.6f}]".format(
                        itr, time.time() - start_time, total_loss, loss
                    )
                )

            if itr % opt.image_log_freq == 0:
                # For image data, add visuals to tb
                output_tensor = torch.sigmoid(output_tensor)
                visuals = [[input_tensor, label_tensor, output_tensor]]
                board_add_images(writer, "grid", visuals, itr)

            writer.add_scalar("total_loss", total_loss, itr)
            writer.add_scalar("loss", loss, itr)

            if itr % opt.save_freq == 0:
                save_checkpoints(opt, itr, u_net)

    print("Training done!")
    if local_rank == 0:
        itr += 1
        save_checkpoints(opt, itr, u_net)

    if opt.do_valid:
        print("validating!")
        total_valid_loss = 0
        for val_i in range(opt.valid_steps):
            data_batch = next(get_valid_data)
            input_tensor, label_tensor = data_batch
            input_tensor = Variable(input_tensor.to(device))
            label_tensor = Variable(label_tensor.to(device))
            with torch.no_grad():
                net_out = net(input_tensor)
            total_valid_loss += loss_CE(net_out, label_tensor).item()

        if local_rank == 0:
            pprint.pprint("[Valid loss-{:.6f}]".format(total_valid_loss))
            writer.add_scalar("valid_loss", total_valid_loss, itr)


if __name__ == "__main__":

    opt = parser()

    if opt.distributed:
        if int(os.environ.get("LOCAL_RANK")) == 0:
            options_printing_saving(opt)
    else:
        options_printing_saving(opt)

    try:
        if opt.distributed:
            print("Initialize Process Group...")
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
            synchronize()

        set_seed(1000)
        training_loop(opt)
        cleanup(opt.distributed)
        print("Exiting..............")

    except KeyboardInterrupt:
        cleanup(opt.distributed)

    except Exception:
        traceback.print_exc(file=sys.stdout)
        cleanup(opt.distributed)
