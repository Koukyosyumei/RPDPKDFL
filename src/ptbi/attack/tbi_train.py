import os

from ..utils.tbi_setup import setup_our_inv_dataloader, setup_tbi_inv_dataloader
from .reconstruction import reconstruct_all_possible_targets


def train_tbi_inv_model(data, device, inv_model, optimizer, criterion):
    x = data[0].to(device)
    y_pred_local = data[1].to(device)

    optimizer.zero_grad()
    x_rec_original = inv_model(y_pred_local.reshape(x.shape[0], -1, 1, 1))
    loss = criterion(x, x_rec_original)
    loss.backward()
    optimizer.step()

    return loss, x, x_rec_original


def train_our_inv_model(data, device, ae, inv_model, optimizer, criterion, gamma=0.1):
    x = data[0].to(device)
    y_pred_local = data[1].to(device)

    optimizer.zero_grad()
    x_rec_original = inv_model(y_pred_local.reshape(x.shape[0], -1, 1, 1))
    loss = criterion(x, x_rec_original) + gamma * criterion(ae(x), x_rec_original)
    loss.backward()
    optimizer.step()

    return loss, x, x_rec_original


def train_our_inv_model_on_logits_dataloader(
    prediction_dataloader, device, ae, inv, inv_optimizer, criterion, gamma=0.1
):
    inv_running_loss = 0
    running_size = 0
    for data in prediction_dataloader:
        loss, x, x_rec = train_our_inv_model(
            data, device, ae, inv, inv_optimizer, criterion, gamma=gamma
        )
        inv_running_loss += loss.item()
        running_size += x.shape[0]

    return inv_running_loss / running_size, x, x_rec


def get_our_inv_train_func(
    client_num,
    is_sensitive_flag,
    local_identities,
    inv_transform,
    return_idx,
    seed,
    batch_size,
    num_workers,
    device,
    inv_tempreature,
    inv_batch_size,
    inv_epoch,
    inv_path_list,
    inv,
    inv_optimizer,
    ae,
    criterion,
    output_dim,
    attack_type,
    id2label,
    output_dir,
    ablation_study,
    gamma=0.1,
):
    def inv_train(api):
        target_client_apis = [
            lambda x_: api.clients[target_client_id](x_).detach()
            for target_client_id in range(client_num)
        ]

        # --- Prepare Public Dataset --- #
        # target_labels = local_identities[target_client_id]

        target_labels = sum(local_identities, [])
        prediction_dataloader = setup_our_inv_dataloader(
            target_labels,
            is_sensitive_flag,
            api,
            target_client_apis,
            inv_transform,
            return_idx,
            seed,
            batch_size,
            num_workers,
            device,
            inv_tempreature,
            inv_batch_size,
        )

        # checkpoint = torch.load(inv_path_list[target_client_id] + ".pth")
        # inv.load_state_dict(checkpoint["model"])
        # inv_optimizer.load_state_dict(checkpoint["optimizer"])

        for i in range(1, inv_epoch + 1):
            (inv_running_loss, _, _) = train_our_inv_model_on_logits_dataloader(
                prediction_dataloader,
                device,
                ae,
                inv,
                inv_optimizer,
                criterion,
                gamma=gamma,
            )

            print(f"inv epoch={i}, inv loss ", inv_running_loss)

            with open(
                os.path.join(output_dir, "inv_result.txt"),
                "a",
                encoding="utf-8",
                newline="\n",
            ) as f:
                f.write(f"{i}, {inv_running_loss}\n")

        # state = {
        #    "model": inv.state_dict(),
        #    "optimizer": inv_optimizer.state_dict(),
        # }
        # torch.save(state, inv_path_list[target_client_id] + ".pth")

        if api.epoch % 2 == 1:
            print("saving the reconstructed images...")
            reconstruct_all_possible_targets(
                attack_type,
                local_identities,
                inv,
                output_dim,
                id2label,
                client_num,
                output_dir,
                device,
                base_name=api.epoch,
            )

    return inv_train


def get_tbi_inv_train_func(
    client_num,
    local_identities,
    inv_transform,
    return_idx,
    seed,
    batch_size,
    num_workers,
    device,
    inv_tempreature,
    inv_batch_size,
    inv_epoch,
    inv_path_list,
    inv,
    inv_optimizer,
    criterion,
    output_dir,
):
    def inv_train(api):

        target_client_apis = [
            lambda x_: api.clients[target_client_id](x_).detach()
            for target_client_id in range(client_num)
        ]

        # --- Prepare Public Dataset --- #
        # target_labels = local_identities[target_client_id]
        target_labels = sum(local_identities, [])
        prediction_dataloader = setup_tbi_inv_dataloader(
            target_labels,
            None,
            api,
            target_client_apis,
            inv_transform,
            return_idx,
            seed,
            batch_size,
            num_workers,
            device,
            inv_tempreature,
            inv_batch_size,
        )

        # checkpoint = torch.load(inv_path_list[target_client_id] + ".pth")
        # inv.load_state_dict(checkpoint["model"])
        # inv_optimizer.load_state_dict(checkpoint["optimizer"])

        for i in range(1, inv_epoch + 1):
            tbi_running_loss = 0
            running_size = 0
            for data in prediction_dataloader:
                loss, x, _ = train_tbi_inv_model(
                    data,
                    device,
                    inv,
                    inv_optimizer,
                    criterion,
                )
                tbi_running_loss += loss.item()
                running_size += x.shape[0]

            tbi_running_loss /= running_size
            print(f"inv epoch={i}, inv loss ", tbi_running_loss)

            with open(
                os.path.join(output_dir, "inv_result.txt"),
                "a",
                encoding="utf-8",
                newline="\n",
            ) as f:
                f.write(f"{i}, {tbi_running_loss}\n")

        # state = {"model": inv.state_dict(), "optimizer": inv_optimizer.state_dict()}
        # torch.save(state, inv_path_list[target_client_id] + ".pth")

    return inv_train
