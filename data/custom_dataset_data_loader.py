import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt):
    dataset = None
    from data.aligned_dataset import AlignedDataset

    dataset = AlignedDataset()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


def CreateValidDataset(opt):
    dataset = None
    from data.valid_dataset import ValidDataset

    dataset = ValidDataset()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return "CustomDatasetDataLoader"

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            sampler=data_sampler(self.dataset, not opt.serial_batches, opt.distributed),
            num_workers=int(opt.nThreads),
            pin_memory=True,
        )

    def get_loader(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)


class ValidDatasetDataLoader(BaseDataLoader):
    def name(self):
        return "ValidDatasetDataLoader"

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateValidDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.Valid_batchSize,
            sampler=data_sampler(self.dataset, False, opt.distributed),
            num_workers=int(opt.nThreads),
            pin_memory=True,
        )

    def get_loader(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.valid_data_size)


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return torch.utils.data.RandomSampler(dataset)

    else:
        return torch.utils.data.SequentialSampler(dataset)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch
