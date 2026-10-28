"""Training and evaluation loops (adapted from the course CNN notebook).

Training order for every batch:
    1. optimizer.zero_grad()   clear old gradients
    2. logits = model(x)       forward pass
    3. loss = criterion(...)   measure the error (on logits, no softmax)
    4. loss.backward()         compute gradients
    5. optimizer.step()        update the weights
"""

import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim


def train_one_epoch(model, loader, criterion, optimizer, device, scheduler=None):
    """One pass over the training data. Returns (mean loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:  # OneCycleLR is updated after every batch
            scheduler.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Loss and accuracy on a labelled loader, in eval mode (BatchNorm/Dropout frozen)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    device,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    scheduler: str | None = None,
    label_smoothing: float = 0.0,
    verbose: bool = True,
):
    """Train with Adam, validate after each epoch and keep the weights with the best validation accuracy.

    scheduler: None (constant learning rate), 'cosine' (CosineAnnealingLR, stepped every epoch)
               or 'onecycle' (OneCycleLR: warm-up then decay, stepped every batch).
    Returns a history dict with train/val loss and accuracy per epoch.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    batch_sched, epoch_sched = None, None
    if scheduler == "onecycle":
        batch_sched = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader)
        )
    elif scheduler == "cosine":  # the scheduler used in the lecture notes (Session 12)
        epoch_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler is not None:
        raise ValueError("scheduler must be None, 'cosine' or 'onecycle'")

    # Validation uses plain cross-entropy so that all models are compared on the same loss.
    val_criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc, best_state, best_epoch = -1.0, None, 0

    for epoch in range(1, epochs + 1):
        start = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, batch_sched
        )
        if epoch_sched is not None:
            epoch_sched.step()
        va_loss, va_acc = evaluate(model, val_loader, val_criterion, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        if va_acc > best_acc:
            best_acc, best_epoch = va_acc, epoch
            best_state = copy.deepcopy(model.state_dict())

        if verbose:
            print(
                f"Epoch {epoch:2d}/{epochs}: train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
                f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} ({time.time() - start:.0f}s)"
            )

    model.load_state_dict(best_state)
    history["best_val_acc"] = best_acc
    history["best_epoch"] = best_epoch
    if verbose:
        print(f"Best validation accuracy {best_acc:.4f} at epoch {best_epoch}")
    return history
