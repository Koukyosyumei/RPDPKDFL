import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from matplotlib import pyplot as plt


def extract_transformd_dataset_from_dataloader(dataloader, return_idx=True):
    x = []
    label = []
    with torch.no_grad():
        if return_idx:
            for _, x_batch, y_batch in dataloader:
                x.append(x_batch)
                label.append(y_batch)
        else:
            for x_batch, y_batch in dataloader:
                x.append(x_batch)
                label.append(y_batch)
    x = torch.cat(x)
    label = torch.cat(label, dim=0).reshape(-1)

    return x, label


def imshow_dataloader(dataloaders, label, figsize=(3, 3), dataset="LAG"):
    num_dataloader = len(dataloaders)
    fig = plt.figure(figsize=figsize)
    for i in range(1, num_dataloader + 1):
        fig.add_subplot(1, num_dataloader, i)
        plt.imshow(
            cv2.cvtColor(
                (
                    dataloaders[i - 1]
                    .dataset.x[dataloaders[i - 1].dataset.y == label]
                    .mean(axis=0)
                    / 255
                ).astype(np.float32),
                cv2.COLOR_BGR2RGB,
            ),
        )
        plt.axis("off")

    plt.show()


def total_variance(x):
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy


def total_variance_numpy(x):
    dx = np.mean(np.abs(x[:, :, :-1] - x[:, :, 1:]))
    dy = np.mean(np.abs(x[:, :-1, :] - x[:, 1:, :]))
    return dx + dy


def grid_plt_show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


def plot_img(x, channel):
    if channel == 1:
        plt.imshow(x.detach().cpu()[0].numpy() * 0.5 + 0.5, cmap="gray")
    else:
        plt.imshow(
            cv2.cvtColor(
                x.detach().cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5,
                cv2.COLOR_BGR2RGB,
            )
        )


def plot_two_images(a, b, channel=3, titles=["a", "b"]):
    fig = plt.figure(figsize=(3, 2))
    fig.add_subplot(1, 2, 1)
    plot_img(a, channel)
    plt.title(titles[0])
    fig.add_subplot(1, 2, 2)
    plot_img(b, channel)
    plt.title(titles[1])
    plt.show()
