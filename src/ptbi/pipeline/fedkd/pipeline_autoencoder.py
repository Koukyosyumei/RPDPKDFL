import os
import random

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

from ...model.invmodel import AE
from ...utils.inv_dataloader import prepare_inv_dataloaders
from ...utils.loss import SSIMLoss


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

    ae = AE().to(device)
    inv_optimizer = torch.optim.Adam(ae.parameters(), lr=inv_lr, weight_decay=0.0001)

    for epoch in range(1, 101):
        running_loss = 0
        for data in inv_dataloader:
            x1 = data[1].to(device)
            x2 = data[2].to(device)

            inv_optimizer.zero_grad()
            x3 = ae(x1)
            loss = criterion(x3, x2)
            loss.backward()
            inv_optimizer.step()

            running_loss += loss.item()

        print(f"epoch={epoch}", running_loss / len(inv_dataloader))

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

        torch.save(x1.detach().cpu(), os.path.join(output_dir, f"img_{epoch}.pth"))
        if epoch % 50 == 0:
            torch.save(ae.state_dict(), os.path.join(output_dir, f"ae_{epoch}.pth"))
