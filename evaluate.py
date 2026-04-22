import numpy as np
import torch

from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack
from utils.metrics import compute_metrics, confusion


ATTACK_MAP = {
    "fgsm": fgsm_attack,
    "pgd": pgd_attack,
}


def _collect_logits_labels(model, loader, device, max_batches=None):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            if max_batches is not None and max_batches > 0 and batch_idx >= max_batches:
                break
            inputs = inputs.to(device)
            logits = model(inputs)
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return logits, labels


def _probs_preds_from_logits(logits: np.ndarray):
    if logits.shape[1] == 1:
        probs = torch.sigmoid(torch.from_numpy(logits)).numpy().squeeze()
        preds = (probs >= 0.5).astype(np.int64)
    else:
        probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        preds = probs.argmax(axis=1)
    return probs, preds


def evaluate_clean(model, loader, device, average="binary", max_batches=None):
    logits, labels = _collect_logits_labels(model, loader, device, max_batches=max_batches)
    probs, preds = _probs_preds_from_logits(logits)
    metrics = compute_metrics(labels, preds, probs, average=average)
    cm = confusion(labels, preds)
    return metrics, cm


def evaluate_under_attack(
    model,
    loader,
    device,
    loss_fn,
    attack_name,
    average="binary",
    epsilon=0.01,
    alpha=0.005,
    iters=5,
    max_batches=None,
):
    if attack_name not in ATTACK_MAP:
        raise ValueError(f"Unsupported attack: {attack_name}")

    attack_fn = ATTACK_MAP[attack_name]
    model.eval()

    all_logits = []
    all_labels = []
    for batch_idx, (inputs, labels) in enumerate(loader):
        if max_batches is not None and max_batches > 0 and batch_idx >= max_batches:
            break
        inputs = inputs.to(device)
        labels = labels.to(device)

        if attack_name == "pgd":
            adv_inputs = attack_fn(
                model,
                loss_fn,
                inputs,
                labels,
                epsilon=epsilon,
                alpha=alpha,
                iters=iters,
            )
        else:
            adv_inputs = attack_fn(
                model,
                loss_fn,
                inputs,
                labels,
                epsilon=epsilon,
            )

        with torch.no_grad():
            logits = model(adv_inputs)
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    probs, preds = _probs_preds_from_logits(logits)
    metrics = compute_metrics(labels, preds, probs, average=average)
    return metrics


def evaluate_robustness(
    model,
    loader,
    device,
    loss_fn,
    average="binary",
    epsilon=0.01,
    alpha=0.005,
    iters=5,
    max_batches=None,
):
    clean_metrics, cm = evaluate_clean(model, loader, device, average=average, max_batches=max_batches)
    fgsm_metrics = evaluate_under_attack(
        model,
        loader,
        device,
        loss_fn,
        attack_name="fgsm",
        average=average,
        epsilon=epsilon,
        alpha=alpha,
        iters=iters,
        max_batches=max_batches,
    )
    pgd_metrics = evaluate_under_attack(
        model,
        loader,
        device,
        loss_fn,
        attack_name="pgd",
        average=average,
        epsilon=epsilon,
        alpha=alpha,
        iters=iters,
        max_batches=max_batches,
    )

    return {
        "clean": clean_metrics,
        "fgsm": fgsm_metrics,
        "pgd": pgd_metrics,
        "accuracy_drop_fgsm": clean_metrics.get("accuracy", 0.0) - fgsm_metrics.get("accuracy", 0.0),
        "accuracy_drop_pgd": clean_metrics.get("accuracy", 0.0) - pgd_metrics.get("accuracy", 0.0),
        "confusion_matrix": cm.tolist(),
    }
