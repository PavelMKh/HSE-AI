import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional, Any
from torch import nn
from torch.utils.data import DataLoader
from IPython.display import clear_output
from tqdm.notebook import tqdm
from model import LanguageModel


sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 15})


def plot_losses(train_losses: List[float], val_losses: List[float]):
    """
    Plot loss and perplexity of train and validation samples
    :param train_losses: list of train losses at each epoch
    :param val_losses: list of validation losses at each epoch
    """
    clear_output()
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    axs[0].plot(range(1, len(train_losses) + 1), train_losses, label='train')
    axs[0].plot(range(1, len(val_losses) + 1), val_losses, label='val')
    axs[0].set_ylabel('loss')

    train_perplexities = [np.exp(loss) for loss in train_losses]
    val_perplexities = [np.exp(loss) for loss in val_losses]

    axs[1].plot(range(1, len(train_perplexities) + 1), train_perplexities, label='train')
    axs[1].plot(range(1, len(val_perplexities) + 1), val_perplexities, label='val')
    axs[1].set_ylabel('perplexity')

    for ax in axs:
        ax.set_xlabel('epoch')
        ax.legend()

    plt.show()


def training_epoch(model: LanguageModel, optimizer: torch.optim.Optimizer, criterion: nn.Module,
                   loader: DataLoader, tqdm_desc: str):
    """
    Process one training epoch
    :param model: language model to train
    :param optimizer: optimizer instance
    :param criterion: loss function class
    :param loader: training dataloader
    :param tqdm_desc: progress bar description
    :return: running train loss
    """
    device = next(model.parameters()).device
    total_loss = 0.0

    model.train()

    for batch_idx, batch_lengths in tqdm(loader, desc=tqdm_desc):
        batch_idx = batch_idx.to(device)

        sorted_lengths, perm_idx = torch.sort(batch_lengths, descending=True)
        sorted_indices = batch_idx[perm_idx]

        outputs = model(sorted_indices, sorted_lengths)

        max_seq_len = min(sorted_lengths.max().item(), model.max_length) - 1

        if max_seq_len < 1:
            continue 

        targets = sorted_indices[:, 1:max_seq_len + 1]
        predictions = outputs[:, :max_seq_len, :]

        mask = (targets != model.pad_id).float()
        total_tokens = mask.sum()


        predictions_flat = predictions.reshape(-1, predictions.size(-1))
        targets_flat = targets.reshape(-1)

        loss_per_token = criterion(predictions_flat, targets_flat)
        masked_loss = (loss_per_token * mask.reshape(-1)).sum() / total_tokens

        optimizer.zero_grad()
        masked_loss.backward()
        optimizer.step()

        total_loss = total_loss + masked_loss.item() * batch_idx.size(0)

    train_loss = total_loss / len(loader.dataset)
    return train_loss


@torch.no_grad()
def validation_epoch(model: LanguageModel, criterion: nn.Module,
                     loader: DataLoader, tqdm_desc: str):
    """
    Process one validation epoch
    :param model: language model to validate
    :param criterion: loss function class
    :param loader: validation dataloader
    :param tqdm_desc: progress bar description
    :return: validation loss
    """
    device = next(model.parameters()).device
    model.eval()
    total_loss = 0.0

    for batch_indices, batch_lengths in tqdm(loader, desc=tqdm_desc):
        batch_indices = batch_indices.to(device)

        sorted_lengths, perm_idx = torch.sort(batch_lengths, descending=True)
        sorted_indices = batch_indices[perm_idx]

        max_seq_len = min(sorted_lengths.max().item(), model.max_length) - 1

        logits = model(sorted_indices, sorted_lengths)
        logits = logits[:, :max_seq_len, :]

        targets = sorted_indices[:, 1:max_seq_len + 1]

        mask = (targets != model.pad_id).float()
        total_tokens = mask.sum()

        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)

        losses = criterion(logits_flat, targets_flat)
        masked_loss = (losses * mask.reshape(-1)).sum() / total_tokens

        total_loss += masked_loss.item() * sorted_indices.size(0)

    average_loss = total_loss / len(loader.dataset)
    return average_loss


def train(model: LanguageModel, optimizer: torch.optim.Optimizer, scheduler: Optional[Any],
          train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, num_examples=5):
    """
    Train language model for several epochs
    :param model: language model to train
    :param optimizer: optimizer instance
    :param scheduler: optional scheduler
    :param train_loader: training dataloader
    :param val_loader: validation dataloader
    :param num_epochs: number of training epochs
    :param num_examples: number of generation examples to print after each epoch
    """
    train_losses, val_losses = [], []
    criterion = nn.CrossEntropyLoss(ignore_index=train_loader.dataset.pad_id)

    for epoch in range(1, num_epochs + 1):
        train_loss = training_epoch(
            model, optimizer, criterion, train_loader,
            tqdm_desc=f'Training {epoch}/{num_epochs}'
        )
        val_loss = validation_epoch(
            model, criterion, val_loader,
            tqdm_desc=f'Validating {epoch}/{num_epochs}'
        )

        if scheduler is not None:
            scheduler.step()

        train_losses += [train_loss]
        val_losses += [val_loss]
        plot_losses(train_losses, val_losses)

        print('Generation examples:')
        for _ in range(num_examples):
            print(model.inference())