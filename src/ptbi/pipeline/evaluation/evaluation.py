import glob
import os

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

from ...utils.utils_data import (
    extract_transformd_dataset_from_dataloader,
    total_variance_numpy,
)


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
    beta=0.5,
    epoch=5,
):
    ssim_list = {
        f"{attack_type}_ssim_private": [],
        f"{attack_type}_ssim_public": [],
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

    for celeb_id in target_ids:
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

        reconstructed_imgs = []
        for i, path in enumerate(
            glob.glob(os.path.join(output_dir, str(epoch) + "_" + str(label) + "_*"))
        ):
            img = cv2.cvtColor(
                np.load(path).transpose(1, 2, 0) * 0.5 + 0.5, cv2.COLOR_BGR2RGB
            )
            reconstructed_imgs.append(img)

        if client_num > 1:
            ssim_matrix = np.zeros((len(reconstructed_imgs), len(reconstructed_imgs)))
            tv_array = np.zeros(len(reconstructed_imgs))
            for i in range(len(reconstructed_imgs)):
                tv_array[i] = total_variance_numpy(reconstructed_imgs[i])
                for j in range(len(reconstructed_imgs)):
                    if i != j:
                        ssim_matrix[i][j] = ssim(
                            reconstructed_imgs[i],
                            reconstructed_imgs[j],
                            multichannel=True,
                            data_range=1,
                        )

            best_img = reconstructed_imgs[
                np.argmin(ssim_matrix.sum(axis=0) / (client_num - 1) + beta * tv_array)
            ]
        else:
            best_img = reconstructed_imgs[0]

        ssim_private_list = [
            ssim(
                cv2.cvtColor(
                    private_dataset_transformed[private_dataset_label == i]
                    .mean(dim=0)
                    .detach()
                    .cpu()
                    .numpy()
                    .transpose(1, 2, 0)
                    * 0.5
                    + 0.5,
                    cv2.COLOR_BGR2RGB,
                ),
                best_img,
                multichannel=True,
            )
            for i in range(num_classes)
        ]
        ssim_public_list = [
            ssim(
                cv2.cvtColor(
                    public_dataset_transformed[public_dataset_label == i]
                    .mean(dim=0)
                    .detach()
                    .cpu()
                    .numpy()
                    .transpose(1, 2, 0)
                    * 0.5
                    + 0.5,
                    cv2.COLOR_BGR2RGB,
                ),
                best_img,
                multichannel=True,
            )
            for i in range(num_classes)
        ]

        best_label = np.nanargmax(ssim_private_list + ssim_public_list)
        ssim_private = ssim_private_list[label]
        ssim_public = ssim_public_list[label]

        result[f"{attack_type}_success"] += label == best_label
        result[f"{attack_type}_too_close_to_public"] += (
            label + num_classes == best_label
        )
        ssim_list[f"{attack_type}_ssim_private"].append(ssim_private)
        ssim_list[f"{attack_type}_ssim_public"].append(ssim_public)

    for k in ssim_list.keys():
        if len(ssim_list[k]) > 0:
            result[k + "_mean"] = np.mean(ssim_list[k])
            result[k + "_std"] = np.std(ssim_list[k])

    return result


def estimate_client_assignment(
    client_num,
    local_dataloaders,
    local_identities,
    id2label,
    output_dir,
    beta=0.5,
    epoch=5,
):

    if client_num == 1:
        return {}

    target_ids = sum(local_identities, [])

    private_dataset_transformed_list = []
    private_dataset_label_list = []
    for i in range(client_num):
        temp_dataset, temp_label = extract_transformd_dataset_from_dataloader(
            local_dataloaders[i], return_idx=True
        )
        private_dataset_transformed_list.append(temp_dataset)
        private_dataset_label_list.append(temp_label)

    result = {}

    for celeb_id in target_ids:
        label = id2label[celeb_id]

        reconstructed_imgs = []
        for i, path in enumerate(
            glob.glob(os.path.join(output_dir, str(epoch) + "_" + str(label) + "_*"))
        ):
            img = cv2.cvtColor(
                np.load(path).transpose(1, 2, 0) * 0.5 + 0.5, cv2.COLOR_BGR2RGB
            )
            reconstructed_imgs.append(img)

        ssim_matrix = np.zeros((len(reconstructed_imgs), len(reconstructed_imgs)))
        tv_array = np.zeros(len(reconstructed_imgs))
        for i in range(len(reconstructed_imgs)):
            tv_array[i] = total_variance_numpy(reconstructed_imgs[i])
            for j in range(len(reconstructed_imgs)):
                if i != j:
                    ssim_matrix[i][j] = ssim(
                        reconstructed_imgs[i],
                        reconstructed_imgs[j],
                        multichannel=True,
                        data_range=1,
                    )

        result[celeb_id] = np.argmin(
            ssim_matrix.sum(axis=0) / (client_num - 1) + beta * tv_array
        )

    return result
