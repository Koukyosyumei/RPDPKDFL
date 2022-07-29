import glob
import random

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from aijack.utils import NumpyDataset, worker_init_fn


def prepare_mnist_dataloaders(
    data_folder="/att",
    client_num=2,
    channel=1,
    batch_size=1,
    seed=42,
    num_workers=2,
    height=64,
    width=64,
    num_classes=10,
    crop=True,
    target_celeblities_num=5,
    blur_strength=10,
):

    dataset_train = torchvision.datasets.MNIST(
        root=data_folder, train=True, download=True
    )

    imgs = dataset_train.train_data.numpy()
    labels = dataset_train.train_labels.numpy()

    local_identities = np.array_split(
        random.sample(np.unique(list(range(10))).tolist(), target_celeblities_num),
        client_num,
    )
    local_identities = [li.tolist() for li in local_identities]

    name_id2client_id = {}
    for client_id, name_id_list in enumerate(local_identities):
        for idx in name_id_list:
            name_id2client_id[idx] = client_id

    X_public_list = []
    y_public_list = []
    X_private_lists = [[] for _ in range(client_num)]
    y_private_lists = [[] for _ in range(client_num)]

    for i in range(num_classes):
        if i in name_id2client_id:
            temp_size = labels[labels == i].shape[0]
            temp_pub_x = imgs[labels == i][: int(temp_size / 2)]
            temp_pub_x[:, 10 : 10 + blur_strength, :] = np.random.uniform(
                0, 255, temp_pub_x[:, 10 : 10 + blur_strength, :].shape
            )
            X_public_list.append(temp_pub_x)
            y_public_list.append(labels[labels == i][: int(temp_size / 2)])
            X_private_lists[name_id2client_id[i]].append(
                imgs[labels == i][int(temp_size / 2) :]
            )
            y_private_lists[name_id2client_id[i]].append(
                labels[labels == i][int(temp_size / 2) :]
            )
        else:
            X_public_list.append(imgs[labels == i])
            y_public_list.append(labels[labels == i])

    X_public = np.stack(X_public_list)
    y_public = np.array(y_public_list)
    X_private_list = [np.stack(x) for x in X_private_lists]
    y_private_list = [np.array(y) for y in y_private_lists]

    transforms_list = [transforms.ToTensor()]
    # if channel == 1:
    #    transforms_list.append(transforms.Grayscale())
    if crop:
        transforms_list.append(transforms.CenterCrop((max(height, width))))
    else:
        transforms_list.append(transforms.Resize((height, width)))

    transforms_list.append(transforms.Normalize((0.5,), (0.5,)))
    # if channel == 1:
    #    transforms_list.append(transforms.Normalize((0.5,), (0.5,)))
    # else:
    #    transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(transforms_list)
    return_idx = True

    public_dataset = NumpyDataset(
        x=X_public,
        y=y_public,
        transform=transform,
        return_idx=return_idx,
    )
    private_dataset_list = [
        NumpyDataset(
            x=X_private_list[i],
            y=y_private_list[i],
            transform=transform,
            return_idx=return_idx,
        )
        for i in range(client_num)
    ]

    g = torch.Generator()
    g.manual_seed(seed)

    print("prepare public dataloader")
    try:
        public_dataloader = torch.utils.data.DataLoader(
            public_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            generator=g,
        )
    except:
        public_dataloader = torch.utils.data.DataLoader(
            public_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
        )

    print("prepare local dataloader")
    try:
        local_dataloaders = [
            torch.utils.data.DataLoader(
                private_dataset_list[i],
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                worker_init_fn=worker_init_fn,
                generator=g,
            )
            for i in range(client_num)
        ]
    except:
        local_dataloaders = [
            torch.utils.data.DataLoader(
                private_dataset_list[i],
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                worker_init_fn=worker_init_fn,
            )
            for i in range(client_num)
        ]

    return public_dataloader, local_dataloaders, local_identities


def prepare_att_dataloaders(
    data_folder="/att",
    client_num=2,
    channel=1,
    batch_size=1,
    seed=42,
    num_workers=2,
    height=64,
    width=64,
    num_classes=1000,
    crop=True,
    target_celeblities_num=100,
    blur_strength=10,
):
    file_paths = glob.glob(f"{data_folder}/*/*.pgm")
    file_paths.sort()
    random.shuffle(file_paths)

    # load images
    imgs = []
    labels = []
    for p in file_paths:
        img = cv2.imread(p, 0)
        label = int(p.replace("\\", "/").split("/")[-2][1:]) - 1
        imgs.append(img)
        labels.append(label)
    imgs = np.stack(imgs)
    labels = np.array(labels)

    local_identities = np.array_split(
        random.sample(np.unique(labels).tolist(), target_celeblities_num), client_num
    )
    local_identities = [li.tolist() for li in local_identities]

    name_id2client_id = {}
    for client_id, name_id_list in enumerate(local_identities):
        for idx in name_id_list:
            name_id2client_id[idx] = client_id

    X_public_list = []
    y_public_list = []
    X_private_lists = [[] for _ in range(client_num)]
    y_private_lists = [[] for _ in range(client_num)]

    label2cnt = {}

    for x, y in zip(imgs, labels):
        if y in name_id2client_id:
            if y not in label2cnt:
                label2cnt[y] = 1
            else:
                label2cnt[y] += 1

            if label2cnt[y] <= 5:
                x[40 : 40 + blur_strength, :] = 0
                X_public_list.append(x)
                # X_public_list.append(cv2.blur(x, (blur_strength, blur_strength)))
                y_public_list.append(y)
            else:
                X_private_lists[name_id2client_id[y]].append(x)
                y_private_lists[name_id2client_id[y]].append(y)

        else:
            X_public_list.append(x)
            y_public_list.append(y)

    X_public = np.stack(X_public_list)
    y_public = np.array(y_public_list)
    X_private_list = [np.stack(x) for x in X_private_lists]
    y_private_list = [np.array(y) for y in y_private_lists]

    transforms_list = [transforms.ToTensor()]
    # if channel == 1:
    #    transforms_list.append(transforms.Grayscale())
    if crop:
        transforms_list.append(transforms.CenterCrop((max(height, width))))
    else:
        transforms_list.append(transforms.Resize((height, width)))

    transforms_list.append(transforms.Normalize((0.5,), (0.5,)))
    # if channel == 1:
    #    transforms_list.append(transforms.Normalize((0.5,), (0.5,)))
    # else:
    #    transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(transforms_list)
    return_idx = True

    public_dataset = NumpyDataset(
        x=X_public,
        y=y_public,
        transform=transform,
        return_idx=return_idx,
    )
    private_dataset_list = [
        NumpyDataset(
            x=X_private_list[i],
            y=y_private_list[i],
            transform=transform,
            return_idx=return_idx,
        )
        for i in range(client_num)
    ]

    g = torch.Generator()
    g.manual_seed(seed)

    print("prepare public dataloader")
    try:
        public_dataloader = torch.utils.data.DataLoader(
            public_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            generator=g,
        )
    except:
        public_dataloader = torch.utils.data.DataLoader(
            public_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
        )

    print("prepare local dataloader")
    try:
        local_dataloaders = [
            torch.utils.data.DataLoader(
                private_dataset_list[i],
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                worker_init_fn=worker_init_fn,
                generator=g,
            )
            for i in range(client_num)
        ]
    except:
        local_dataloaders = [
            torch.utils.data.DataLoader(
                private_dataset_list[i],
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                worker_init_fn=worker_init_fn,
            )
            for i in range(client_num)
        ]

    return public_dataloader, local_dataloaders, local_identities


def prepare_lfw_dataloaders(
    data_folder="/content",
    client_num=2,
    channel=1,
    batch_size=1,
    seed=42,
    num_workers=2,
    height=64,
    width=64,
    num_classes=1000,
    crop=True,
    target_celeblities_num=100,
):
    """Prepare dataloaders for LFW dataset.

    Args:
        data_folder (str, optional): a path to the data folder. Defaults to "/content".
        client_num (int, optional): the number of clients. Defaults to 2.
        channel (int, optional): the number of channes. Defaults to 1.
        batch_size (int, optional): batch size for training. Defaults to 1.
        seed (int, optional): seed of randomness. Defaults to 42.
        num_workers (int, optional): the number of workers. Defaults to 2.
        height (int, optional): height of images. Defaults to 64.
        width (int, optional): width of images. Defaults to 64.
        num_classes (int, optional): the number of classes. Defaults to 1000.
        crop (bool, optional): crop the image to (height, width) if true. Defaults to True.
        target_celeblities_num (int, optional): the number of target labels. Defaults to 100.

    Returns:
        a tuple of public dataloader, a list of local dataloaders, test dataloader,
        and the list of target labels.
    """
    nomask_path_list = glob.glob(f"{data_folder}/lfw-align-128/*/*")
    nomask_path_list.sort()
    mask_path_list = glob.glob(f"{data_folder}/lfw-align-128-masked/*/*")
    mask_path_list.sort()

    path_list = []
    name_list = []
    ismask_list = []
    for mask_path in mask_path_list:
        name = mask_path.split("/")[-2]
        file_name = mask_path.split("/")[-1]
        nomask_path = f"{data_folder}/lfw-align-128/{name}/{file_name}"
        name_list.append(name)
        if random.random() > 0.5:
            path_list.append(mask_path)
            ismask_list.append(1)
        else:
            path_list.append(nomask_path)
            ismask_list.append(0)

    df = pd.DataFrame(columns=["name", "path", "ismask"])
    df["name"] = name_list
    df["path"] = path_list
    df["ismask"] = ismask_list

    top_identities = (
        df.groupby("name")
        .count()
        .sort_values("ismask", ascending=False)
        .index[:num_classes]
    )
    df["top"] = df["name"].apply(lambda x: x in top_identities)
    df = df[df["top"]]

    name_with_both_types_of_images = []
    for name in df["name"].unique():
        if df[df["name"] == name].groupby("ismask").count().shape[0] > 1:
            name_with_both_types_of_images.append(name)

    name2id = {name: i for i, name in enumerate(df["name"].unique())}

    local_identities_names = np.array_split(
        random.sample(name_with_both_types_of_images, target_celeblities_num),
        client_num,
    )
    local_identities = [
        [name2id[name] for name in name_list] for name_list in local_identities_names
    ]

    name_id2client_id = {}
    for client_id, name_id_list in enumerate(local_identities):
        for idx in name_id_list:
            name_id2client_id[idx] = client_id

    X_public_list = []
    y_public_list = []
    X_private_lists = [[] for _ in range(client_num)]
    y_private_lists = [[] for _ in range(client_num)]

    for name, path, ismask in zip(
        df["name"].to_list(), df["path"].to_list(), df["ismask"].to_list()
    ):
        if name2id[name] in name_id2client_id and ismask == 0:
            X_private_lists[name_id2client_id[name2id[name]]].append(cv2.imread(path))
            y_private_lists[name_id2client_id[name2id[name]]].append(name2id[name])
        else:
            X_public_list.append(cv2.imread(path))
            y_public_list.append(name2id[name])

    X_public = np.stack(X_public_list)
    y_public = np.array(y_public_list)
    X_private_list = [np.stack(x) for x in X_private_lists]
    y_private_list = [np.array(y) for y in y_private_lists]

    transforms_list = [transforms.ToTensor()]
    if channel == 1:
        transforms_list.append(transforms.Grayscale())
    if crop:
        transforms_list.append(transforms.CenterCrop((max(height, width))))
    else:
        transforms_list.append(transforms.Resize((height, width)))
    if channel == 1:
        transforms_list.append(transforms.Normalize((0.5,), (0.5,)))
    else:
        transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(transforms_list)
    return_idx = True

    public_dataset = NumpyDataset(
        x=X_public,
        y=y_public,
        transform=transform,
        return_idx=return_idx,
    )
    private_dataset_list = [
        NumpyDataset(
            x=X_private_list[i],
            y=y_private_list[i],
            transform=transform,
            return_idx=return_idx,
        )
        for i in range(client_num)
    ]

    g = torch.Generator()
    g.manual_seed(seed)

    print("prepare public dataloader")
    try:
        public_dataloader = torch.utils.data.DataLoader(
            public_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            generator=g,
        )
    except:
        public_dataloader = torch.utils.data.DataLoader(
            public_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
        )

    print("prepare local dataloader")
    try:
        local_dataloaders = [
            torch.utils.data.DataLoader(
                private_dataset_list[i],
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                worker_init_fn=worker_init_fn,
                generator=g,
            )
            for i in range(client_num)
        ]
    except:
        local_dataloaders = [
            torch.utils.data.DataLoader(
                private_dataset_list[i],
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                worker_init_fn=worker_init_fn,
            )
            for i in range(client_num)
        ]

    return public_dataloader, local_dataloaders, local_identities


def prepare_lag_dataloaders(
    data_folder="../input/large-agegap",
    client_num=2,
    batch_size=1,
    channel=1,
    seed=42,
    num_workers=2,
    height=64,
    width=64,
    num_classes=1010,
    crop=True,
    target_celeblities_num=100,
):
    """Prepare dataloaders for LAG dataset.

    Args:
        data_folder (str, optional): a path to the data folder. Defaults to "/content".
        client_num (int, optional): the number of clients. Defaults to 2.
        channel (int, optional): the number of channes. Defaults to 1.
        batch_size (int, optional): batch size for training. Defaults to 1.
        seed (int, optional): seed of randomness. Defaults to 42.
        num_workers (int, optional): the number of workers. Defaults to 2.
        height (int, optional): height of images. Defaults to 64.
        width (int, optional): width of images. Defaults to 64.
        num_classes (int, optional): the number of classes. Defaults to 1000.
        crop (bool, optional): crop the image to (height, width) if true. Defaults to True.
        target_celeblities_num (int, optional): the number of target labels. Defaults to 100.

    Returns:
        a tuple of public dataloader, a list of local dataloaders, test dataloader,
        and the list of target labels.
    """
    paths = glob.glob(f"{data_folder}/*")
    paths.sort()

    name_list = []
    path_list = []
    ay_list = []

    for p in paths:
        name = p.split("/")[-1]
        if name == "README.txt":
            continue
        a_paths = glob.glob(f"{p}/*.*")
        y_paths = glob.glob(f"{p}/y/*.*")

        name_list += [name for _ in range(len(a_paths))]
        name_list += [name for _ in range(len(y_paths))]
        path_list += a_paths
        path_list += y_paths
        ay_list += [1 for _ in range(len(a_paths))]
        ay_list += [0 for _ in range(len(y_paths))]
    df = pd.DataFrame(columns=["name", "path", "ay"])
    df["name"] = name_list
    df["path"] = path_list
    df["ay"] = ay_list

    top_identities = (
        df.groupby("name")
        .count()
        .sort_values("ay", ascending=False)
        .index[:num_classes]
    )
    df["top"] = df["name"].apply(lambda x: x in top_identities)
    df = df[df["top"]]

    print(df.head())

    unique_name_list = []
    unique_name_min_img_num = []

    for name in df["name"].unique():
        unique_name_list.append(name)
        unique_name_min_img_num.append(
            df[df["name"] == name].groupby("ay").count().min()["name"]
        )
    unique_name_list = np.array(unique_name_list)
    name2id = {name: i for i, name in enumerate(unique_name_list)}
    unique_name_min_img_num = np.array(unique_name_min_img_num)

    local_identities = random.sample(
        list(range(len(unique_name_list))), target_celeblities_num
    )
    local_identities = np.array_split(local_identities, client_num)
    local_identities = [id_list.tolist() for id_list in local_identities]

    print("target identities is ", local_identities)

    alloc = [-1 for _ in range(df.shape[0])]
    for j, (ay, name) in enumerate(zip(df["ay"].tolist(), df["name"].tolist())):
        if ay == 1:
            for i in range(client_num):
                if name2id[name] in local_identities[i]:
                    alloc[j] = i + 1
                    break
        if alloc[j] == -1:
            alloc[j] = 0
    df["alloc"] = alloc

    print(df[df["alloc"] == 0]["path"].tolist()[:10])

    X_public = np.stack([cv2.imread(p) for p in df[df["alloc"] == 0]["path"].tolist()])
    y_public = np.array([name2id[n] for n in df[df["alloc"] == 0]["name"].tolist()])
    X_private_list = [
        np.stack([cv2.imread(p) for p in df[df["alloc"] == i + 1]["path"].tolist()])
        for i in range(client_num)
    ]
    y_private_list = [
        np.array([name2id[n] for n in df[df["alloc"] == i + 1]["name"].tolist()])
        for i in range(client_num)
    ]

    transforms_list = [transforms.ToTensor()]
    if channel == 1:
        transforms_list.append(transforms.Grayscale())
    if crop:
        transforms_list.append(transforms.CenterCrop((max(height, width))))
    else:
        transforms_list.append(transforms.Resize((height, width)))
    if channel == 1:
        transforms_list.append(transforms.Normalize((0.5,), (0.5,)))
    else:
        transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(transforms_list)
    return_idx = True

    public_dataset = NumpyDataset(
        x=X_public,
        y=y_public,
        transform=transform,
        return_idx=return_idx,
    )
    private_dataset_list = [
        NumpyDataset(
            x=X_private_list[i],
            y=y_private_list[i],
            transform=transform,
            return_idx=return_idx,
        )
        for i in range(client_num)
    ]

    g = torch.Generator()
    g.manual_seed(seed)

    print("prepare public dataloader")
    try:
        public_dataloader = torch.utils.data.DataLoader(
            public_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            generator=g,
        )
    except:
        public_dataloader = torch.utils.data.DataLoader(
            public_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
        )

    print("prepare local dataloader")
    try:
        local_dataloaders = [
            torch.utils.data.DataLoader(
                private_dataset_list[i],
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                worker_init_fn=worker_init_fn,
                generator=g,
            )
            for i in range(client_num)
        ]
    except:
        local_dataloaders = [
            torch.utils.data.DataLoader(
                private_dataset_list[i],
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                worker_init_fn=worker_init_fn,
            )
            for i in range(client_num)
        ]

    return public_dataloader, local_dataloaders, local_identities


def prepare_dataloaders(dataset_name, *args, **kwargs):
    """Return dataloaders

    Args:
        dataset_name (str): name of dataset (`LAG` or `LFW`)

    Raises:
        NotImplementedError: if name is not LAG or LFW.

    Returns:
        a tuple of public dataloader, a list of local dataloaders, test dataloader,
        and the list of target labels.
    """
    if dataset_name == "LAG":
        return prepare_lag_dataloaders(*args, **kwargs)
    elif dataset_name == "LFW":
        return prepare_lfw_dataloaders(*args, **kwargs)
    elif dataset_name == "AT&T":
        return prepare_att_dataloaders(*args, **kwargs)
    elif dataset_name == "MNIST":
        return prepare_mnist_dataloaders(*args, **kwargs)
    else:
        raise NotImplementedError(f"{dataset_name} is not supported")
