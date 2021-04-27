import random
import albumentations as A


class apply_augmentation(object):
    def __init__(self):

        self.augs_dict = {
            # compression
            "image_compression": 0.01,
            "downscale": 0.01,
            # blur noise
            "blur": 0.05,
            "motion_blur": 0.1,
            "gaussian_blur": 0.01,
            "gaussian_noise": 0.01,
            "iso_noise": 0.2,  # camera sensor noise
            "optical_distortion": 0.1,
            # color
            "channel_dropout": 0.05,
            "channel_shuffle": 0.01,
            "color_jitter": 0.2,
            "bightness": 0.4,
            "contrast": 0.2,
            "bightness_contrast": 0.3,
            "rgb_shift": 0.01,
            "sharpen": 0.01,
            # natural
            "bight_dark": 0.1,
            "fog": 0.1,
            "rain": 0.1,
            "snow": 0.05,
            "shadow": 0.1,
            "SunFlare": 0.2,
        }
        augs_dict = self.augs_dict

        self.compression_augs = [
            A.ImageCompression(quality_lower=75, p=augs_dict["image_compression"]),
            A.Downscale(scale_min=0.7, scale_max=0.999, p=augs_dict["downscale"]),
        ]

        self.blur_noise_augs = [
            A.Blur(p=augs_dict["blur"]),
            A.MotionBlur(p=augs_dict["motion_blur"]),
            A.GaussianBlur(p=augs_dict["gaussian_blur"]),
            A.GaussNoise(p=augs_dict["gaussian_noise"]),
            A.ISONoise(p=augs_dict["iso_noise"]),
            A.OpticalDistortion(p=augs_dict["optical_distortion"]),
        ]
        self.blur_noise_weight = [1, 1.5, 0.7, 0.7, 2, 1.5]

        self.color_augs = [
            A.ChannelDropout(p=augs_dict["channel_dropout"]),
            A.ChannelShuffle(p=augs_dict["channel_shuffle"]),
            A.ColorJitter(p=augs_dict["color_jitter"]),
            A.RandomBrightness(p=augs_dict["bightness"]),
            A.RandomContrast(p=augs_dict["contrast"]),
            A.RandomBrightnessContrast(p=augs_dict["bightness_contrast"]),
            A.RGBShift(p=augs_dict["rgb_shift"]),
        ]
        self.color_weight = [0.7, 0.7, 1, 1.5, 1, 1.5, 0.7]

        self.natural_augs = [
            A.RandomFog(p=augs_dict["fog"]),
            A.RandomRain(p=augs_dict["rain"]),
            # A.RandomShadow(p=augs_dict["shadow"]),
            A.RandomSnow(p=augs_dict["snow"]),
            A.RandomSunFlare(
                num_flare_circles_lower=4,
                num_flare_circles_upper=8,
                src_radius=200,
                p=augs_dict["SunFlare"],
            ),
        ]
        self.natural_weights = [1, 0.7, 1, 1, 1.5]

    def __call__(self, opt):
        aug_list = []

        if opt.color_aug:
            aug_list += self.color_augs

        if opt.natural_aug:
            aug_list += self.natural_augs

        if opt.blur_noise_aug:
            aug_list += self.blur_noise_augs

        if opt.compression_aug:
            aug_list += self.compression_augs

        return aug_list
