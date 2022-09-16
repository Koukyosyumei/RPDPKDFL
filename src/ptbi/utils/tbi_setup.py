import os
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from aijack.utils import NumpyDataset, worker_init_fn
from torch import nn

from ..model.invmodel import get_invmodel_class


def setup_training_based_inversion(
    attack_type,
    invmodel_type,
    num_classes,
    client_num,
    inv_lr,
    device,
    config_dataset,
    temp_dir,
    ablation_study,
):
    temp_path_list = []
    for i in range(client_num):
        if attack_type == "ptbi":
            if ablation_study not in [3, 4]:
                inv = get_invmodel_class(invmodel_type)(
                    input_dim=num_classes * 2 + 1,
                    output_shape=(
                        config_dataset["channel"],
                        config_dataset["height"],
                        config_dataset["width"],
                    ),
                    channel=config_dataset["channel"],
                ).to(device)
            elif ablation_study == 4:
                inv = get_invmodel_class(invmodel_type)(
                    input_dim=num_classes * 2,
                    output_shape=(
                        config_dataset["channel"],
                        config_dataset["height"],
                        config_dataset["width"],
                    ),
                    channel=config_dataset["channel"],
                ).to(device)
            else:
                inv = get_invmodel_class(invmodel_type)(
                    input_dim=num_classes,
                    output_shape=(
                        config_dataset["channel"],
                        config_dataset["height"],
                        config_dataset["width"],
                    ),
                    channel=config_dataset["channel"],
                ).to(device)
        elif attack_type == "tbi":
            inv = get_invmodel_class(invmodel_type)(
                input_dim=num_classes,
                output_shape=(
                    config_dataset["channel"],
                    config_dataset["height"],
                    config_dataset["width"],
                ),
                channel=config_dataset["channel"],
            ).to(device)
        inv_optimizer = torch.optim.Adam(
            inv.parameters(), lr=inv_lr, weight_decay=0.0001
        )
        # inv_optimizer_finetune = torch.optim.Adam(
        #    inv.parameters(), lr=inv_lr / 5, weight_decay=0.0001
        # )
        state = {
            "model": inv.state_dict(),
            "optimizer": inv_optimizer.state_dict(),
            # "finetune_optimizer": inv_optimizer_finetune.state_dict(),
        }
        temp_path = os.path.join(temp_dir, f"client_{i}")
        torch.save(state, temp_path + ".pth")
        temp_path_list.append(temp_path)

    return temp_path_list, inv, inv_optimizer, None


def setup_tbi_optimizers(dataset_name, config_dataset):
    transforms_list = [transforms.ToTensor()]
    if dataset_name not in ["AT&T", "MNIST"]:
        if "channel" not in config_dataset or config_dataset["channel"] != 3:
            transforms_list.append(transforms.Grayscale())
    if "crop" in config_dataset and config_dataset["crop"]:
        transforms_list.append(
            transforms.CenterCrop(
                (max(config_dataset["height"], config_dataset["width"]))
            )
        )
    else:
        transforms_list.append(
            transforms.Resize((config_dataset["height"], config_dataset["width"]))
        )
    if "channel" not in config_dataset or config_dataset["channel"] == 1:
        transforms_list.append(transforms.Normalize((0.5,), (0.5,)))
    else:
        transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    inv_transform = transforms.Compose(transforms_list)
    return inv_transform


def setup_inv_dataloader(
    target_labels,
    is_sensitive_flag,
    api,
    target_client_api,
    inv_transform,
    return_idx,
    seed,
    batch_size,
    num_workers,
    device,
    inv_tempreature,
    inv_batch_size,
):

    inv_trainset = NumpyDataset(
        x=api.public_dataloader.dataset.x,
        y=api.public_dataloader.dataset.y,
        transform=inv_transform,
        return_idx=return_idx,
    )

    g = torch.Generator()
    g.manual_seed(seed)
    inv_public_dataloader = torch.utils.data.DataLoader(
        inv_trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=g,
    )

    # --- Receive logits --- #
    public_x_list = []
    y_pred_server_list = []
    y_pred_local_list = []
    flag_list = []
    for data in inv_public_dataloader:
        idx = data[0]
        if is_sensitive_flag is not None:
            flag_list.append(torch.Tensor(is_sensitive_flag[idx]))
        x = data[1].to(device).detach()
        y_pred_server = torch.softmax(api.server(x) / inv_tempreature, dim=-1).detach()
        y_pred_local = torch.softmax(
            target_client_api(x) / inv_tempreature, dim=-1
        ).detach()
        public_x_list.append(x.cpu())
        y_pred_server_list.append(y_pred_server.cpu())
        y_pred_local_list.append(y_pred_local.cpu())
    public_x_tensor = torch.cat(public_x_list)
    y_pred_server_tensor = torch.cat(y_pred_server_list)
    y_pred_local_tensor = torch.cat(y_pred_local_list)
    flag_tensor = None
    if is_sensitive_flag is not None:
        flag_tensor = torch.cat(flag_list).reshape(-1, 1)

    if is_sensitive_flag is not None:
        prediction_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                public_x_tensor, y_pred_server_tensor, y_pred_local_tensor, flag_tensor
            ),
            batch_size=inv_batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            generator=g,
        )
    else:
        prediction_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                public_x_tensor, y_pred_server_tensor, y_pred_local_tensor
            ),
            batch_size=inv_batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            generator=g,
        )

    return prediction_dataloader


def setup_paired_inv_dataloader(
    target_labels,
    is_sensitive_flag,
    api,
    target_client_api,
    inv_transform,
    return_idx,
    seed,
    batch_size,
    num_workers,
    device,
    inv_tempreature,
    inv_batch_size,
):
    inv_trainset = NumpyDataset(
        x=api.public_dataloader.dataset.x,
        y=api.public_dataloader.dataset.y,
        transform=inv_transform,
        return_idx=return_idx,
    )

    g = torch.Generator()
    g.manual_seed(seed)
    inv_public_dataloader = torch.utils.data.DataLoader(
        inv_trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=g,
    )

    # --- Receive logits --- #
    public_x_list = []
    y_local_list = []
    y_pred_local_list = []
    is_sensitive_flag_list = []
    flag_list = []

    for data in inv_public_dataloader:
        idx = data[0]
        # if is_sensitive_flag is not None:
        #    flag_list.append(torch.Tensor(is_sensitive_flag[idx]))
        x = data[1].to(device).detach()
        y_pred_local = torch.softmax(
            target_client_api(x) / inv_tempreature, dim=-1
        ).detach()
        public_x_list.append(x.cpu())
        y_pred_local_list.append(y_pred_local.cpu())

        y_local_list.append(data[2].detach().numpy())
        is_sensitive_flag_list.append(is_sensitive_flag[idx])

    public_x_tensor = torch.cat(public_x_list)
    y_pred_local_tensor = torch.cat(y_pred_local_list)
    y_local_array = np.concatenate(y_local_list)
    is_sensitive_flag_array = np.concatenate(is_sensitive_flag_list)

    sensitive_idx = np.where(is_sensitive_flag_array == 1)[0]
    nonsensitive_idx = np.where(is_sensitive_flag_array == 0)[0]

    X_paired_nonsensitive_list = []
    X_paired_sensitive_list = []
    y_paired_list = []

    for target_y in range(1000):
        if target_y in target_labels:
            continue
        y_idx = np.where(y_local_array == target_y)[0]
        y_sensitive_idx = list(set(list(y_idx)) & set(list(sensitive_idx)))
        y_nonsensitive_idx = list(set(list(y_idx)) & set(list(nonsensitive_idx)))

        pairs = sum(
            [[(ys, yn) for yn in y_nonsensitive_idx] for ys in y_sensitive_idx], []
        )
        pairs = random.sample(pairs, min(50, len(pairs)))

        for pair in pairs:
            X_paired_nonsensitive_list.append(public_x_tensor[[pair[0]]])
            X_paired_sensitive_list.append(public_x_tensor[[pair[1]]])
            y_paired_list.append(
                y_pred_local_tensor[[pair[0]]] / 2 + y_pred_local_tensor[[pair[1]]] / 2
            )

    x_paired_nonsensitive = torch.cat(X_paired_nonsensitive_list)
    x_paired_sensitive = torch.cat(X_paired_sensitive_list)
    y_paired = torch.cat(y_paired_list)

    print(x_paired_nonsensitive.shape)
    print(x_paired_sensitive.shape)
    print(y_paired.shape)

    prediction_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            x_paired_nonsensitive, x_paired_sensitive, y_paired
        ),
        batch_size=inv_batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=g,
    )

    return prediction_dataloader


class SubLM(nn.Module):
    def __init__(self, input_dim=3 * 128 * 64, hidden_dim=2000, output_dim=1000):
        super(SubLM, self).__init__()
        self.fla = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fla(x)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x


def setup_paired_inversion(
    attack_type,
    invmodel_type,
    num_classes,
    client_num,
    inv_lr,
    device,
    config_dataset,
    temp_dir,
    ablation_study,
):
    temp_path_list = []
    for i in range(client_num):
        sub = SubLM().to(device)

        sub_optimizer = torch.optim.Adam(
            sub.parameters(), lr=inv_lr, weight_decay=0.0001
        )
        # inv_optimizer_finetune = torch.optim.Adam(
        #    inv.parameters(), lr=inv_lr / 5, weight_decay=0.0001
        # )
        state = {
            "model": sub.state_dict(),
            "optimizer": sub_optimizer.state_dict(),
            # "finetune_optimizer": inv_optimizer_finetune.state_dict(),
        }
        temp_path = os.path.join(temp_dir, f"client_{i}")
        torch.save(state, temp_path + ".pth")
        temp_path_list.append(temp_path)

    return temp_path_list, sub, sub_optimizer, None
