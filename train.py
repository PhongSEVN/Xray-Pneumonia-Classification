import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import shutil
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, Resize, ToTensor, RandomAffine

from configs.train_config import BATCH_SIZE, NUM_WORKERS, EPOCHS, DATA_DIR, IMG_SIZE
from dataset.dataset import Xray
from models.CNN_model import CNN_model


from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import Normalize

def get_args():
    parser = ArgumentParser(description='CNN Training')
    parser.add_argument("--epochs","-e", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size","-b", type=int, default=BATCH_SIZE, help="Number of batch size")
    parser.add_argument("--image_size","-i", type=int, default=IMG_SIZE, help="Number of image size")
    parser.add_argument("--root","-r", type=str, default=DATA_DIR, help="Root of dataset")
    parser.add_argument("--logging","-l", type=str, default="tensorboard", help="Log training")
    parser.add_argument("--model","-m", type=str, default="trained_model", help="Model to use")
    parser.add_argument("--checkpoint","-c", type=str, default=None, help="checkpoint")

    args = parser.parse_args()
    return args

def plot_confusion_matrix(writer, cm, class_name, epoch):
    """Plot confusion matrix with modern styling"""
    plt.style.use('seaborn-v0_8-darkgrid')
    figure, ax = plt.subplots(figsize=(8, 8))
    
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(f"Confusion Matrix - Epoch {epoch + 1}", fontsize=14, fontweight='bold', pad=15)
    
    cbar = figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Count', rotation=-90, va="bottom", fontsize=10)
    
    tick_marks = np.arange(len(class_name))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_name, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(class_name, fontsize=11)

    cm_norm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    threshold = cm.max() / 2

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.0%})", 
                   ha="center", va="center", color=color, fontsize=12, fontweight='bold')
    
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    figure.tight_layout()
    writer.add_figure("Charts/Confusion_Matrix", figure, epoch)
    plt.close(figure)


def plot_training_curves(writer, train_losses, val_accuracies, learning_rates, epoch):
    """Plot training curves with modern styling for TensorBoard"""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs_range = range(1, len(train_losses) + 1)
    
    # Plot 1: Training Loss
    ax1 = axes[0]
    ax1.plot(epochs_range, train_losses, 'o-', color='#e74c3c', linewidth=2, markersize=4, label='Train Loss')
    ax1.fill_between(epochs_range, train_losses, alpha=0.3, color='#e74c3c')
    ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax1.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation Accuracy
    ax2 = axes[1]
    ax2.plot(epochs_range, val_accuracies, 's-', color='#27ae60', linewidth=2, markersize=4, label='Val Accuracy')
    ax2.fill_between(epochs_range, val_accuracies, alpha=0.3, color='#27ae60')
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax2.set_title('Validation Accuracy', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Learning Rate
    ax3 = axes[2]
    ax3.plot(epochs_range, learning_rates, 'd-', color='#3498db', linewidth=2, markersize=4, label='Learning Rate')
    ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Learning Rate', fontsize=11, fontweight='bold')
    ax3.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax3.set_yscale('log')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    fig.suptitle(f'Training Progress - Epoch {epoch + 1}', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    
    writer.add_figure("Charts/Training_Curves", fig, epoch)
    plt.close(fig)


def calculate_class_weights(dataset):
    labels = dataset.label_path
    class_counts = np.bincount(labels)
    total_samples = len(labels)

    class_weights = total_samples / (len(class_counts) * class_counts)
    return torch.FloatTensor(class_weights)


if __name__ == '__main__':

    args = get_args()

    if torch.cuda.is_available():
        print("Using GPU")
        device = torch.device('cuda')
    else:
        print("Using CPU")
        device = torch.device('cpu')

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = Compose([
        RandomAffine(degrees=(-5, 5),
                     translate=(0.15, 0.15),
                     shear=(-5, 5),
                     scale=(0.8, 1.2),
                     ),
        Resize((args.image_size,args.image_size)),
        ToTensor(),
        Normalize(mean, std)
    ])
    test_transform = Compose([
        Resize((args.image_size,args.image_size)),
        ToTensor(),
        Normalize(mean, std)
    ])

    train_dataset = Xray(root=args.root,train=True, transform=train_transform)
    test_dataset = Xray(root=args.root,train=False, transform=test_transform)

    train_loader = DataLoader(dataset=train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=NUM_WORKERS,drop_last=True)
    test_loader = DataLoader(dataset=test_dataset,batch_size=args.batch_size,shuffle=False,num_workers=NUM_WORKERS)

    class_weights = calculate_class_weights(train_dataset).to(device)

    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging)

    os.makedirs(args.model, exist_ok=True)

    writter = SummaryWriter(args.logging)

    model = CNN_model().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        best_accuracy = checkpoint.get("best_accuracy", 0)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        start_epoch = 0
        best_accuracy = 0

    # Lists to track metrics for plotting
    train_losses_history = []
    val_accuracies_history = []
    learning_rates_history = []

    num_iters = len(train_loader)

    print("\n" + "="*50)
    print(f"{'Epoch':<10} | {'Train Loss':<15} | {'Val Accuracy':<15}")
    print("-" * 50)

    for epoch in range(start_epoch,args.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm.tqdm(train_loader, colour='green', leave=False)
        for iter, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            
            output = model(images)
            loss = criterion(output, labels)
            
            progress_bar.set_description(
                f"Epoch [{epoch + 1}/{args.epochs}] | Loss: {loss.item():.4f}"
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        writter.add_scalar("Metrics/Train_Loss", epoch_loss, epoch)
        writter.add_scalar("Metrics/Learning_Rate", current_lr, epoch)
        
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

            cm = confusion_matrix(all_labels, all_preds)
            plot_confusion_matrix(writter, cm, class_name=test_dataset.categories, epoch=epoch)
            accuracy = accuracy_score(all_labels, all_preds)
            
            # Print row for the table
            print(f"{epoch + 1:<10} | {epoch_loss:<15.4f} | {accuracy:<15.4f}")
            
            # Log validation accuracy
            writter.add_scalar("Metrics/Val_Accuracy", accuracy, epoch)
            
            # Track history for custom plots
            train_losses_history.append(epoch_loss)
            val_accuracies_history.append(accuracy)
            learning_rates_history.append(current_lr)
            
            # Plot custom training curves
            plot_training_curves(writter, train_losses_history, val_accuracies_history, learning_rates_history, epoch)
            
            # Update scheduler
            scheduler.step(accuracy)
            
            checkpoint = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": epoch_loss,
                "best_accuracy": best_accuracy
            }
            torch.save(checkpoint, f"{args.model}/last_cnn.pt")
            
            if accuracy > best_accuracy:
                print(f"Best model saved (Acc: {accuracy:.4f})")
                best_accuracy = accuracy
                checkpoint["best_accuracy"] = best_accuracy
                torch.save(checkpoint, f"{args.model}/best_cnn.pt")
    
    print("="*50)
    writter.close()
