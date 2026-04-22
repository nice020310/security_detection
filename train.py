import argparse
import csv
import json
import os
from datetime import datetime

import numpy as np
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim

from utils.dataset import load_csv_dataset, make_dataloaders
from evaluate import evaluate_clean, evaluate_robustness

from models.cnn import CNN1DClassifier
from models.bilstm import BiLSTMClassifier
from models.transformer import TransformerClassifier

from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack

MODEL_MAP = {
    "cnn": CNN1DClassifier,
    "bilstm": BiLSTMClassifier,
    "transformer": TransformerClassifier,
}

ATTACK_MAP = {
    "fgsm": fgsm_attack,
    "pgd": pgd_attack,
}


def parse_args():
    parser = argparse.ArgumentParser("Security Detection Training")
    parser.add_argument("--csv", type=str, required=True, help="CSV 数据路径")
    parser.add_argument("--label", type=str, required=True, help="标签列名")
    parser.add_argument("--model", type=str, default="cnn", choices=list(MODEL_MAP.keys()), help="模型类型")
    parser.add_argument("--compare_models", nargs="*", default=[], help="用于对比训练的模型列表")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--attack", type=str, default=None, choices=["fgsm", "pgd", "none"], help="对抗训练的攻击方法")
    parser.add_argument("--epsilon", type=float, default=0.01, help="攻击强度 epsilon")
    parser.add_argument("--alpha", type=float, default=0.005, help="PGD 步长 alpha")
    parser.add_argument("--iters", type=int, default=5, help="PGD 迭代次数")
    parser.add_argument("--average", type=str, default="binary", help="指标聚合，binary/macro/weighted")
    parser.add_argument("--robust_eval", action="store_true", help="是否做鲁棒性评估")
    parser.add_argument("--robust_eval_batches", type=int, default=0, help="鲁棒性评估使用的测试集 batch 数量，0 表示全部")
    parser.add_argument("--enable_tsne", action="store_true", help="是否计算 t-SNE（耗时）")
    parser.add_argument("--tsne_max_points", type=int, default=300, help="t-SNE 最大采样点数")
    parser.add_argument("--save_model", action="store_true", help="是否保存模型")
    parser.add_argument("--export_results", action="store_true", help="是否导出结果")
    parser.add_argument("--output_dir", type=str, default="outputs", help="模型与结果输出目录")
    return parser.parse_args()


def create_model(model_name: str, num_features: int, num_classes: int):
    cls = MODEL_MAP[model_name]
    return cls(num_features=num_features, num_classes=num_classes)


def _compute_loss(loss_fn, logits, y):
    if logits.shape[1] == 1:
        return loss_fn(logits.squeeze(1), y.float())
    return loss_fn(logits, y)


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = _compute_loss(loss_fn, logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def train_one_epoch_adversarial(model, loader, loss_fn, optimizer, device, attack_name, epsilon, alpha, iters):
    model.train()
    total_loss = 0.0
    attack_fn = ATTACK_MAP[attack_name]

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        if attack_name == "pgd":
            adv_inputs = attack_fn(model, loss_fn, X, y, epsilon=epsilon, alpha=alpha, iters=iters)
        else:
            adv_inputs = attack_fn(model, loss_fn, X, y, epsilon=epsilon)

        model.train()
        optimizer.zero_grad()
        logits = model(adv_inputs)
        loss = _compute_loss(loss_fn, logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _save_model(model, output_dir, model_name, timestamp):
    model_dir = os.path.join(output_dir, "models")
    _ensure_dir(model_dir)
    model_path = os.path.abspath(os.path.join(model_dir, f"{model_name}_{timestamp}.pt"))
    state_dict = model.state_dict()
    with open(model_path, "wb") as f:
        torch.save(state_dict, f)
    return model_path


def _export_results(output_dir, timestamp, payload):
    result_dir = os.path.join(output_dir, "results")
    _ensure_dir(result_dir)

    json_path = os.path.abspath(os.path.join(result_dir, f"result_{timestamp}.json"))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    csv_path = os.path.abspath(os.path.join(result_dir, f"comparison_{timestamp}.csv"))
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "clean_accuracy", "clean_f1", "fgsm_accuracy", "pgd_accuracy", "roc_auc"])
        for row in payload.get("comparison", []):
            writer.writerow([
                row.get("model"),
                row.get("clean_accuracy"),
                row.get("clean_f1"),
                row.get("fgsm_accuracy"),
                row.get("pgd_accuracy"),
                row.get("roc_auc"),
            ])

    return {"result_json": json_path, "comparison_csv": csv_path}


def _build_tsne_points(splits, max_points=500):
    X_test = splits["test"]["X"]
    y_test = splits["test"]["y"]
    n = len(X_test)
    if n == 0:
        return []
    n_use = min(max_points, n)
    if n_use < n:
        _, idx = train_test_split(np.arange(n), test_size=n_use, random_state=42, stratify=y_test)
        X = X_test[idx]
        y = y_test[idx]
    else:
        X = X_test
        y = y_test
    perplexity = min(30, max(2, n_use // 5))
    if perplexity >= n_use:
        perplexity = max(1, n_use - 1)
    tsne = TSNE(
        n_components=2,
        random_state=42,
        learning_rate="auto",
        init="pca",
        perplexity=perplexity,
    )
    coords = tsne.fit_transform(X)
    points = []
    for i in range(len(coords)):
        points.append({
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "label": int(y[i]),
        })
    return points


def _build_risk_profile(history, clean_metrics, robustness):
    clean_acc = float(clean_metrics.get("accuracy", 0.0) or 0.0)
    robust_drop = 0.0
    if robustness:
        robust_drop = max(
            float(robustness.get("accuracy_drop_fgsm", 0.0) or 0.0),
            float(robustness.get("accuracy_drop_pgd", 0.0) or 0.0),
        )

    base_risk = 100.0 * (1.0 - clean_acc)
    robust_penalty = 100.0 * robust_drop * 0.7
    score = max(0.0, min(100.0, base_risk + robust_penalty))

    trend = []
    for item in history:
        epoch = int(item.get("epoch", 0))
        val_acc = item.get("val_accuracy")
        if val_acc is None:
            val_acc = clean_acc
        trend.append({"epoch": epoch, "risk": max(0.0, min(100.0, 100.0 * (1.0 - float(val_acc))))})

    return {
        "score": round(score, 2),
        "level": "high" if score >= 66 else ("medium" if score >= 33 else "low"),
        "trend": trend,
    }


def _train_single_model(model_name, args, loaders, num_features, num_classes, device, callback, stop_event, timestamp):
    if callback:
        callback(f"Building model: {model_name}")

    model = create_model(model_name, num_features, num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss() if num_classes > 1 else nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    use_attack = None if args.attack in (None, "none") else args.attack

    history = []
    for epoch in range(args.epochs):
        if stop_event and stop_event.is_set():
            if callback:
                callback("Training stopped by user.")
            return None

        if use_attack is None:
            loss = train_one_epoch(model, loaders["train"], loss_fn, optimizer, device)
            log_msg = f"[{model_name}] [Train] epoch={epoch+1}/{args.epochs}, loss={loss:.4f}"
        else:
            loss = train_one_epoch_adversarial(
                model,
                loaders["train"],
                loss_fn,
                optimizer,
                device,
                attack_name=use_attack,
                epsilon=args.epsilon,
                alpha=args.alpha,
                iters=args.iters,
            )
            log_msg = f"[{model_name}] [AdvTrain:{use_attack}] epoch={epoch+1}/{args.epochs}, loss={loss:.4f}"

        val_metrics, _ = evaluate_clean(model, loaders["val"], device, average=args.average)
        history_item = {
            "epoch": epoch + 1,
            "model": model_name,
            "loss": float(loss),
            "val_accuracy": float(val_metrics.get("accuracy", 0.0)),
            "val_f1": float(val_metrics.get("f1", 0.0)),
        }
        history.append(history_item)
        log_msg = (
            f"{log_msg}, val_acc={history_item['val_accuracy']:.4f}, "
            f"val_f1={history_item['val_f1']:.4f}"
        )
        print(log_msg)
        if callback:
            callback(log_msg, data=history_item)

    clean_metrics, cm = evaluate_clean(model, loaders["test"], device, average=args.average)
    robustness = None
    if args.robust_eval:
        if callback:
            callback(f"[{model_name}] Running robustness evaluation (FGSM/PGD)...")
        robustness = evaluate_robustness(
            model,
            loaders["test"],
            device,
            loss_fn,
            average=args.average,
            epsilon=args.epsilon,
            alpha=args.alpha,
            iters=args.iters,
            max_batches=getattr(args, "robust_eval_batches", 0),
        )

    model_path = None
    if args.save_model:
        model_path = _save_model(model, args.output_dir, model_name, timestamp)
        if callback:
            callback(f"[{model_name}] Model saved: {model_path}")

    result = {
        "model": model_name,
        "history": history,
        "clean_metrics": clean_metrics,
        "confusion_matrix": cm.tolist(),
        "robustness": robustness,
        "risk": _build_risk_profile(history, clean_metrics, robustness),
        "model_path": model_path,
    }
    return result


def run_training(args, callback=None, stop_event=None):
    splits = load_csv_dataset(args.csv, label_column=args.label)
    num_features = splits["train"]["X"].shape[1]
    y_train = splits["train"]["y"]
    if len(y_train) == 0:
        raise ValueError("Training set is empty")
    num_classes = int(np.max(y_train)) + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not hasattr(args, "compare_models") or args.compare_models is None:
        args.compare_models = []
    if not hasattr(args, "robust_eval"):
        args.robust_eval = False
    if not hasattr(args, "enable_tsne"):
        args.enable_tsne = True
    if not hasattr(args, "tsne_max_points"):
        args.tsne_max_points = 300
    if not hasattr(args, "save_model"):
        args.save_model = False
    if not hasattr(args, "export_results"):
        args.export_results = False
    if not hasattr(args, "output_dir") or not args.output_dir:
        args.output_dir = "outputs"

    args.output_dir = os.path.abspath(args.output_dir)
    loaders = make_dataloaders(splits, batch_size=args.batch_size)
    models_to_run = [m for m in [args.model] + list(args.compare_models) if m in MODEL_MAP]
    models_to_run = list(dict.fromkeys(models_to_run))

    if callback:
        callback(f"Device: {device}")
        callback(f"Features: {num_features}, Classes: {num_classes}")
        callback(f"Running models: {', '.join(models_to_run)}")

    model_results = []
    for model_name in models_to_run:
        if callback:
            callback(f"Starting training for model: {model_name}")
        result = _train_single_model(
            model_name=model_name,
            args=args,
            loaders=loaders,
            num_features=num_features,
            num_classes=num_classes,
            device=device,
            callback=callback,
            stop_event=stop_event,
            timestamp=timestamp,
        )
        if result is None:
            return None
        model_results.append(result)
        partial_comparison = []
        for r in model_results:
            clean = r.get("clean_metrics", {})
            robust = r.get("robustness") or {}
            fgsm_acc = (robust.get("fgsm") or {}).get("accuracy")
            pgd_acc = (robust.get("pgd") or {}).get("accuracy")
            partial_comparison.append({
                "model": r["model"],
                "clean_accuracy": clean.get("accuracy"),
                "clean_f1": clean.get("f1"),
                "fgsm_accuracy": fgsm_acc,
                "pgd_accuracy": pgd_acc,
                "roc_auc": clean.get("roc_auc"),
            })
        if callback:
            callback(
                f"Completed model: {model_name}",
                data={
                    "results": {
                        "primary_model": args.model,
                        "comparison": partial_comparison,
                        "models": model_results,
                        "tsne": [],
                        "artifacts": {},
                    }
                },
            )

    comparison = []
    for r in model_results:
        clean = r.get("clean_metrics", {})
        robust = r.get("robustness") or {}
        fgsm_acc = (robust.get("fgsm") or {}).get("accuracy")
        pgd_acc = (robust.get("pgd") or {}).get("accuracy")
        comparison.append({
            "model": r["model"],
            "clean_accuracy": clean.get("accuracy"),
            "clean_f1": clean.get("f1"),
            "fgsm_accuracy": fgsm_acc,
            "pgd_accuracy": pgd_acc,
            "roc_auc": clean.get("roc_auc"),
        })

    final_payload = {
        "primary_model": args.model,
        "comparison": comparison,
        "models": model_results,
        "tsne": [],
        "artifacts": {},
    }

    if getattr(args, "enable_tsne", True):
        if callback:
            callback("Computing t-SNE...", data={"results": final_payload})
        try:
            final_payload["tsne"] = _build_tsne_points(splits, max_points=int(getattr(args, "tsne_max_points", 300)))
            if callback:
                callback("t-SNE completed.", data={"results": final_payload})
        except Exception as e:
            if callback:
                callback(f"t-SNE failed: {str(e)}", data={"results": final_payload})

    if args.export_results:
        final_payload["artifacts"] = _export_results(args.output_dir, timestamp, final_payload)
        if callback:
            callback(f"Results exported: {final_payload['artifacts']}")

    if callback:
        callback("Training and evaluation completed.", data={"results": final_payload})

    return final_payload


if __name__ == "__main__":
    args = parse_args()
    run_training(args)
