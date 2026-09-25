from pathlib import Path
import argparse as arg
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm

from tools.ascad_loader import load_dataset, dissemble_data_dict
from tools.loadData import get_labels
from tools.model_zoo_pytorch import build_cnn_best


def parse_args():
    parser=arg.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="Output/cnn_pytorch")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--target_byte", type=int, default=2)
    parser.add_argument("--leakage_model", type=str, choices=["HW", "ID"], default="HW")
    parser.add_argument("--n_traces", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss=0.0
    correct=0
    total=0

    progress=tqdm(dataloader, desc="Train", leave=False, dynamic_ncols=True)

    for traces, labels in progress:
        traces=traces.to(device)
        labels=labels.to(device)

        optimizer.zero_grad()

        logits=model(traces)
        loss=criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss+=loss.item()*traces.size(0)

        preds=torch.argmax(logits, dim=1)
        correct+=(preds==labels).sum().item()
        total+=labels.size(0)

        progress.set_postfix(loss=f"{loss.item():.6f}", acc=f"{correct/total:.4f}")

    avg_loss=running_loss/total
    accuracy=correct/total

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss=0.0
    correct=0
    total=0

    with torch.no_grad():
        for traces, labels in dataloader:
            traces=traces.to(device)
            labels=labels.to(device)

            logits=model(traces)
            loss=criterion(logits, labels)

            running_loss+=loss.item()*traces.size(0)

            preds=torch.argmax(logits, dim=1)
            correct+=(preds==labels).sum().item()
            total+=labels.size(0)

    avg_loss=running_loss/total
    accuracy=correct/total

    return avg_loss, accuracy


def main():
    args=parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if torch.cuda.is_available():
        device=torch.device("cuda")
    elif torch.backends.mps.is_available():
        device=torch.device("mps")
    else:
        device=torch.device("cpu")

    print("Device:", device)

    # ============================================================
    # Output paths
    # ============================================================

    run_root=Path(args.output_root)/f"N_{args.n_traces}"
    ckpt_path=run_root/"ckpt"/"cnn_best.pt"
    indices_path=run_root/"selected_indices.npy"
    log_path=run_root/"training_log.npz"

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Load profiling data
    # ============================================================

    data_dict=load_dataset(data_path=args.data_path, which_one="train")
    traces, _, plaintext, key=dissemble_data_dict(data_dict=data_dict, tracewindow=(0,700), which_one="train")

    print("Full profiling traces:", traces.shape)
    print("Full profiling plaintext:", plaintext.shape)

    if args.n_traces > len(traces):
        raise ValueError(f"n_traces={args.n_traces} exceeds profiling dataset size={len(traces)}")

    # ============================================================
    # Select exactly N profiling traces
    # ============================================================

    selected_indices=np.random.choice(len(traces), size=args.n_traces, replace=False)

    x_n=traces[selected_indices]
    plaintext_n=plaintext[selected_indices]

    np.save(indices_path, selected_indices)

    print("Selected N traces:", x_n.shape)
    print("Saved selected indices to:", indices_path)

    # ============================================================
    # Generate labels
    # ============================================================

    key_byte=key[args.target_byte]
    labels_n=get_labels(plaintext_n, key_byte, args.target_byte, args.leakage_model)

    if args.leakage_model=="HW":
        num_classes=9
    else:
        num_classes=256

    print("Labels shape:", labels_n.shape)
    print("Classes:", sorted(np.unique(labels_n).tolist()))

    # ============================================================
    # Match Keras validation_split=0.1 behavior
    # ============================================================

    val_size=int(len(x_n)*args.val_split)
    train_size=len(x_n)-val_size

    x_train=x_n[:train_size]
    y_train=labels_n[:train_size]

    x_val=x_n[train_size:]
    y_val=labels_n[train_size:]

    print("Train shape:", x_train.shape)
    print("Validation shape:", x_val.shape)

    # ============================================================
    # Convert to tensors
    # ============================================================

    x_train=torch.as_tensor(x_train, dtype=torch.float32)
    x_val=torch.as_tensor(x_val, dtype=torch.float32)

    if x_train.ndim==2:
        x_train=x_train.unsqueeze(-1)

    if x_val.ndim==2:
        x_val=x_val.unsqueeze(-1)

    y_train=torch.as_tensor(y_train, dtype=torch.long)
    y_val=torch.as_tensor(y_val, dtype=torch.long)

    train_dataset=TensorDataset(x_train, y_train)
    val_dataset=TensorDataset(x_val, y_val)

    train_loader=DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader=DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # ============================================================
    # Build CNN classifier
    # ============================================================

    model=build_cnn_best(input_shape=(x_n.shape[1],1), emb_size=num_classes, classification=True)
    model=model.to(device)

    criterion=nn.CrossEntropyLoss()

    optimizer=torch.optim.RMSprop(model.parameters(), lr=args.learning_rate, alpha=0.9, eps=1e-7, momentum=0.0, centered=False)

    # ============================================================
    # Train
    # ============================================================

    best_val_acc=-1.0

    train_loss_log=[]
    train_acc_log=[]
    val_loss_log=[]
    val_acc_log=[]

    epoch_bar=tqdm(range(args.epochs), desc="CNN", dynamic_ncols=True)

    for epoch in epoch_bar:
        train_loss, train_acc=train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc=evaluate(model, val_loader, criterion, device)

        train_loss_log.append(train_loss)
        train_acc_log.append(train_acc)
        val_loss_log.append(val_loss)
        val_acc_log.append(val_acc)

        epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}", train_acc=f"{train_acc:.4f}", val_loss=f"{val_loss:.4f}", val_acc=f"{val_acc:.4f}")

        if val_acc > best_val_acc:
            old_best=best_val_acc
            best_val_acc=val_acc

            torch.save(model.state_dict(), ckpt_path)

            tqdm.write(f"Best val accuracy improved: {old_best:.6f} -> {best_val_acc:.6f}")
            tqdm.write(f"Saved checkpoint to: {ckpt_path}")

    # ============================================================
    # Restore best model
    # ============================================================

    state_dict=torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)

    # ============================================================
    # Save training logs
    # ============================================================

    np.savez(log_path, train_loss=np.asarray(train_loss_log), train_acc=np.asarray(train_acc_log), val_loss=np.asarray(val_loss_log), val_acc=np.asarray(val_acc_log))

    print("Saved training log to:", log_path)
    print("CNN training finished.")


if __name__=="__main__":
    main()