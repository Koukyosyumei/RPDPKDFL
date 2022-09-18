import glob
import os

import cv2
import numpy as np
import torch
import tqdm
from skimage.metrics import structural_similarity as ssim

from ...utils.loss import SSIMLoss
from ...utils.utils_data import extract_transformd_dataset_from_dataloader


def evaluate_ssim(
    x_rec,
    private_dataset_transformed,
    public_dataset_transformed,
    private_dataset_label,
    public_dataset_label,
    num_classes,
    target_label,
    channel=1,
):
    multichannel = channel == 3
    if multichannel:
        ssim_private_list = [
            ssim(
                private_dataset_transformed[private_dataset_label == i]
                .mean(dim=0)
                .detach()
                .cpu()
                .numpy()
                .transpose(1, 2, 0),
                x_rec,
                multichannel=multichannel,
            )
            for i in range(num_classes)
        ]
        ssim_public_list = [
            ssim(
                public_dataset_transformed[public_dataset_label == i]
                .mean(dim=0)
                .detach()
                .cpu()
                .numpy()
                .transpose(1, 2, 0),
                x_rec,
                multichannel=multichannel,
            )
            for i in range(num_classes)
        ]
    else:
        ssim_private_list = [
            ssim(
                private_dataset_transformed[private_dataset_label == i]
                .mean(dim=0)[0]
                .detach()
                .cpu()
                .numpy(),
                x_rec[:, :, 0],
                multichannel=multichannel,
            )
            for i in range(num_classes)
        ]
        ssim_public_list = [
            ssim(
                public_dataset_transformed[public_dataset_label == i]
                .mean(dim=0)[0]
                .detach()
                .cpu()
                .numpy(),
                x_rec[:, :, 0],
                multichannel=multichannel,
            )
            for i in range(num_classes)
        ]

    best_label = np.nanargmax(ssim_private_list + ssim_public_list)
    ssim_private = ssim_private_list[target_label]
    ssim_public = ssim_public_list[target_label]
    return (
        target_label == best_label,
        target_label + num_classes == best_label,
        ssim_private,
        ssim_public,
    )


def evaluation_full(
    client_num,
    num_classes,
    public_dataloader,
    local_dataloaders,
    local_identities,
    id2label,
    attack_type,
    output_dir,
    epoch=5,
    device="cuda:0",
):
    print("evaluating ...")

    ssim_list = {
        f"{attack_type}_ssim_private": [],
        f"{attack_type}_ssim_public": [],
    }
    mse_list = {
        f"{attack_type}_mse_private": [],
        f"{attack_type}_mse_public": [],
    }
    result = {
        f"{attack_type}_success": 0,
        f"{attack_type}_too_close_to_public": 0,
    }

    target_ids = sum(local_identities, [])

    (
        public_dataset_transformed,
        public_dataset_label,
    ) = extract_transformd_dataset_from_dataloader(public_dataloader, return_idx=True)

    private_dataset_transformed_list = []
    private_dataset_label_list = []
    for i in range(client_num):
        temp_dataset, temp_label = extract_transformd_dataset_from_dataloader(
            local_dataloaders[i], return_idx=True
        )
        private_dataset_transformed_list.append(temp_dataset)
        private_dataset_label_list.append(temp_label)
    private_dataset_transformed = torch.cat(private_dataset_transformed_list)
    private_dataset_label = torch.cat(private_dataset_label_list)

    ssim = SSIMLoss()

    for celeb_id in tqdm.tqdm(target_ids):
        label = id2label[celeb_id]

        np.save(
            os.path.join(output_dir, "private_" + str(label)),
            cv2.cvtColor(
                private_dataset_transformed[private_dataset_label == label]
                .mean(dim=0)
                .detach()
                .cpu()
                .numpy()
                .transpose(1, 2, 0)
                * 0.5
                + 0.5,
                cv2.COLOR_BGR2RGB,
            ),
        )

        np.save(
            os.path.join(output_dir, "public_" + str(label)),
            cv2.cvtColor(
                public_dataset_transformed[public_dataset_label == label]
                .mean(dim=0)
                .detach()
                .cpu()
                .numpy()
                .transpose(1, 2, 0)
                * 0.5
                + 0.5,
                cv2.COLOR_BGR2RGB,
            ),
        )
        temp_path = glob.glob(
            os.path.join(output_dir, str(epoch) + "_" + str(label) + "_*")
        )[0]

        best_img_tensor = torch.Tensor(np.load(temp_path)).to(device)
        best_img_tensor_batch = torch.stack(
            [best_img_tensor.clone() for _ in range(num_classes)]
        ).to(device)
        best_img_tensor_batch = best_img_tensor_batch * 0.5 + 0.5

        private_data = torch.stack(
            [
                private_dataset_transformed[private_dataset_label == i].mean(dim=0)
                for i in range(num_classes)
            ]
        ).to(device)
        private_data = private_data * 0.5 + 0.5

        public_data = torch.stack(
            [
                public_dataset_transformed[public_dataset_label == i].mean(dim=0)
                for i in range(num_classes)
            ]
        ).to(device)
        public_data = public_data * 0.5 + 0.5

        # evaluation on ssim

        ssim_private_list = np.zeros(num_classes)
        ssim_public_list = np.zeros(num_classes)

        for idxs in np.array_split(list(range(num_classes)), 2):
            ssim_private_list[idxs] = (
                ssim(best_img_tensor_batch[idxs], private_data[idxs], False)
                .detach()
                .cpu()
                .numpy()
            )
            ssim_public_list[idxs] = (
                ssim(best_img_tensor_batch[idxs], public_data[idxs], False)
                .detach()
                .cpu()
                .numpy()
            )

        best_label = np.nanargmax(ssim_private_list + ssim_public_list)
        ssim_private = ssim_private_list[label]
        ssim_public = ssim_public_list[label]

        result[f"{attack_type}_success"] += label == best_label
        result[f"{attack_type}_too_close_to_public"] += (
            label + num_classes == best_label
        )
        ssim_list[f"{attack_type}_ssim_private"].append(ssim_private)
        ssim_list[f"{attack_type}_ssim_public"].append(ssim_public)

        # evaluation on mse

        mse_list[f"{attack_type}_mse_private"].append(
            torch.nn.functional.mse_loss(best_img_tensor, private_data[label]).item()
        )
        mse_list[f"{attack_type}_mse_public"].append(
            torch.nn.functional.mse_loss(best_img_tensor, public_data[label]).item()
        )

    for k in ssim_list.keys():
        if len(ssim_list[k]) > 0:
            result[k + "_mean"] = np.mean(ssim_list[k])
            result[k + "_std"] = np.std(ssim_list[k])

    for k in mse_list.keys():
        if len(mse_list[k]) > 0:
            result[k + "_mean"] = np.mean(mse_list[k])
            result[k + "_std"] = np.std(mse_list[k])

    return result
