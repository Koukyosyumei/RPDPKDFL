import copy
import os
import pickle
import random

import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from PIL import Image

from ...attack.confidence import get_alpha, get_pi
from ...attack.reconstruction import (
    reconstruct_all_possible_targets,
    reconstruct_private_data_and_quick_evaluate,
)
from ...attack.styletransfer import (
    cnn_normalization_mean,
    cnn_normalization_std,
    run_style_transfer,
)
from ...attack.tbi_train import (
    get_inv_train_fn_ablation_3,
    get_inv_train_fn_ptbi,
    get_inv_train_fn_tbi,
)
from ...model.invmodel import AE
from ...model.model import get_model_class
from ...utils.dataloader import prepare_dataloaders
from ...utils.fedkd_setup import get_fedkd_api
from ...utils.inv_dataloader import prepare_inv_dataloaders
from ...utils.tbi_setup import setup_tbi_optimizers, setup_training_based_inversion
from ..evaluation.evaluation import evaluation_full


def unloader(img):
    image = img.clone()
    image = image.cpu()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)
    return transforms.ToPILImage()(image)


def ae_attack_fedkd(
    fedkd_type="FedGEMS",
    model_type="LM",
    invmodel_type="InvCNN",
    attack_type="ptbi",
    dataset="AT&T",
    client_num=2,
    batch_size=4,
    inv_batch_size=1,
    lr=0.01,
    num_classes=20,
    num_communication=5,
    seed=42,
    num_workers=2,
    inv_epoch=10,
    inv_lr=0.003,
    inv_tempreature=1.0,
    use_finetune=True,
    inv_pj=0.5,
    beta=0.5,
    evaluation_type="quick",
    ablation_study=0,
    config_fedkd=None,
    config_dataset=None,
    config_attack_nes=None,
    output_dir="",
    temp_dir="./",
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

    return_idx = True

    # --- Setup device --- #
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    print("device is ", device)

    # --- Setup DataLoaders --- #
    (
        public_train_dataloader,
        local_train_dataloaders,
        test_dataloader,
        local_identities,
        is_sensitive_flag,
    ) = prepare_dataloaders(
        dataset_name=dataset,
        client_num=client_num,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
        num_classes=num_classes,
        **config_dataset,
    )

    inv_dataloader = prepare_inv_dataloaders(
        dataset_name=dataset,
        client_num=client_num,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
        num_classes=num_classes,
        **config_dataset,
    )

    ae = AE().to(device)
    inv_optimizer = torch.optim.Adam(ae.parameters(), lr=0.001, weight_decay=0.0001)

    for epoch in range(5):
        running_loss = 0
        for data in inv_dataloader:
            x1 = data[1].to(device)
            x2 = data[2].to(device)

            inv_optimizer.zero_grad()
            x3 = ae(x1)
            loss = torch.nn.MSELoss()(x3, x2)
            loss.backward()
            inv_optimizer.step()

            running_loss += loss.item()

        print(f"epoch={epoch}", running_loss)

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
