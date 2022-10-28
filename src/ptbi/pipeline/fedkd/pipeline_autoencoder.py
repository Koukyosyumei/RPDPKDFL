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
        self.direction = "AtoB"
        self.model = "cycle_gan"
        self.input_nc = 3
        self.output_nc = 3
        self.ngf = 64
        self.ndf = 64
        self.netD = "basic"
        self.netG = "resnet_3blocks"
        self.n_layers_D = 3
        self.norm = "instance"
        self.init_type = "normal"
        self.init_gain = 0.02
        self.no_dropout = True
        self.n_epochs = 100
        self.n_epochs_decay = 100
        self.beta1 = 0.5
        self.lr = 0.0002
        self.gan_mode = "lsgan"
        self.pool_size = 50
        self.lr_policy = "linear"
        self.lr_decay_iters = 50
        self.checkpoints_dir = "./"
        self.gpu_ids = [0]

        self.epoch_count = 1

        self.load_iter = 50
        self.continue_train = False

        self.isTrain = True

        self.lambda_A = 10.0
        self.lambda_B = 10.0
        self.lambda_identity = 0.5

        self.verbose = 2

        self.initialized = True


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

    opt = BaseOptions()
    opt.checkpoints_dir = output_dir

    model = CycleGANModel(opt)
    model.setup(opt)

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
