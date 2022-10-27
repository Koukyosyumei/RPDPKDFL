import argparse
import os
import random

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

from ...model.cycle_gan_model import CycleGANModel
from ...utils.inv_dataloader import prepare_inv_dataloaders
from ...utils.loss import SSIMLoss


class BaseOptions:
    """This class defines options used during both training and test time.
    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        parser.add_argument(
            "--direction", type=str, default="AtoB", help="AtoB or BtoA"
        )
        # model parameters
        parser.add_argument(
            "--model",
            type=str,
            default="cycle_gan",
            help="chooses which model to use. [cycle_gan | pix2pix | test | colorization]",
        )
        parser.add_argument(
            "--input_nc",
            type=int,
            default=3,
            help="# of input image channels: 3 for RGB and 1 for grayscale",
        )
        parser.add_argument(
            "--output_nc",
            type=int,
            default=3,
            help="# of output image channels: 3 for RGB and 1 for grayscale",
        )
        parser.add_argument(
            "--ngf",
            type=int,
            default=64,
            help="# of gen filters in the last conv layer",
        )
        parser.add_argument(
            "--ndf",
            type=int,
            default=64,
            help="# of discrim filters in the first conv layer",
        )
        parser.add_argument(
            "--netD",
            type=str,
            default="basic",
            help="specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator",
        )
        parser.add_argument(
            "--netG",
            type=str,
            default="resnet_3blocks",
            help="specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]",
        )
        parser.add_argument(
            "--n_layers_D", type=int, default=3, help="only used if netD==n_layers"
        )
        parser.add_argument(
            "--norm",
            type=str,
            default="instance",
            help="instance normalization or batch normalization [instance | batch | none]",
        )
        parser.add_argument(
            "--init_type",
            type=str,
            default="normal",
            help="network initialization [normal | xavier | kaiming | orthogonal]",
        )
        parser.add_argument(
            "--init_gain",
            type=float,
            default=0.02,
            help="scaling factor for normal, xavier and orthogonal.",
        )
        parser.add_argument(
            "--no_dropout", action="store_true", help="no dropout for the generator"
        )
        parser.add_argument(
            "--n_epochs",
            type=int,
            default=100,
            help="number of epochs with the initial learning rate",
        )
        parser.add_argument(
            "--n_epochs_decay",
            type=int,
            default=100,
            help="number of epochs to linearly decay learning rate to zero",
        )
        parser.add_argument(
            "--beta1", type=float, default=0.5, help="momentum term of adam"
        )
        parser.add_argument(
            "--lr", type=float, default=0.0002, help="initial learning rate for adam"
        )
        parser.add_argument(
            "--gan_mode",
            type=str,
            default="lsgan",
            help="the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.",
        )
        parser.add_argument(
            "--pool_size",
            type=int,
            default=50,
            help="the size of image buffer that stores previously generated images",
        )
        parser.add_argument(
            "--lr_policy",
            type=str,
            default="linear",
            help="learning rate policy. [linear | step | plateau | cosine]",
        )
        parser.add_argument(
            "--lr_decay_iters",
            type=int,
            default=50,
            help="multiply by a gamma every lr_decay_iters iterations",
        )
        parser.add_argument("--checkpoints_dir", type=str, default="./")
        self.initialized = True
        return parser


def unloader(img):
    image = img.clone()
    image = image.cpu()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)
    return transforms.ToPILImage()(image)


def ae_attack_fedkd(
    dataset="AT&T",
    client_num=2,
    batch_size=4,
    num_classes=20,
    inv_lr=0.00003,
    seed=42,
    num_workers=2,
    loss_type="mse",
    config_dataset=None,
    output_dir="",
):
    # --- Fix seed --- #
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    try:
        torch.use_deterministic_algorithms(True)
    except:
        print("torch.use_deterministic_algorithms is not available")

    try:
        torch.backends.cudnn.benchmark = False
    except:
        print("torch.backends.cudnn.benchmark is not available")

    # --- Setup device --- #
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    print("device is ", device)

    # --- Setup DataLoaders --- #

    inv_dataloader = prepare_inv_dataloaders(
        dataset_name=dataset,
        client_num=client_num,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
        num_classes=num_classes,
        **config_dataset,
    )

    # --- Setup loss function --- #
    if loss_type == "mse":
        criterion = torch.nn.MSELoss()
    elif loss_type == "ssim":
        criterion = SSIMLoss()
    else:
        raise NotImplementedError(
            f"{loss_type} is not supported. We currently support `mse` or `ssim`."
        )

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser = BaseOptions().initialize(parser)

    parser.checkpoints_dir = output_dir

    model = CycleGANModel(parser).to(device)

    for epoch in range(1, 51):
        model.update_learning_rate()
        for data in inv_dataloader:
            x1 = data[1].to(device)
            x2 = data[2].to(device)

            model.set_input(
                {"A": data[1], "B": data[2]}
            )  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()

        x3 = model.netG_A(x1[[0]])

        figure = plt.figure()
        figure.add_subplot(1, 3, 1)
        plt.imshow(
            cv2.cvtColor(
                x1[0].detach().cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5,
                cv2.COLOR_BGR2RGB,
            )
        )
        figure.add_subplot(1, 3, 2)
        plt.imshow(
            cv2.cvtColor(
                x2[0].detach().cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5,
                cv2.COLOR_BGR2RGB,
            )
        )
        figure.add_subplot(1, 3, 3)
        plt.imshow(
            cv2.cvtColor(
                x3[0].detach().cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5,
                cv2.COLOR_BGR2RGB,
            )
        )
        plt.savefig(f"{epoch}.png")

        if epoch % 50 == 0:
            model.save_networks(epoch)
