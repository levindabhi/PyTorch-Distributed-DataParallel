import os.path as osp
import os


class parser(object):
    def __init__(self):
        self.name = "testing"
        self.image_dir = 
        self.mask_dir = 
        self.isTrain = True
        self.distributed = False

        # Mean std params
        self.mean = 0.5
        self.std = 0.5

        self.batchSize = 2  # 12
        self.nThreads = 2  # 3
        self.max_dataset_size = float("inf")

        self.do_valid = False
        if self.do_valid:
            self.valid_image_dir = ""
            self.valid_mask_dir = ""
            self.valid_freq = 1000
            self.Valid_batchSize = 16
            self.valid_data_size = 500
            self.valid_steps = int(self.valid_data_size / self.Valid_batchSize)

        self.serial_batches = False
        self.continue_train = True
        if self.continue_train:
            self.net_checkpoint = 

        self.color_aug = True
        self.natural_aug = True
        self.blur_noise_aug = True
        self.compression_aug = False

        self.save_freq = 1000
        self.print_freq = 10
        self.image_log_freq = 100

        self.iter = 100000
        self.lr = 0.0005
        self.clip_grad = 3

        self.logs_dir = osp.join("logs", self.name)
        self.save_dir = osp.join("results", self.name)
