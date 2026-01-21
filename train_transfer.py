import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, Resize, ToTensor, RandomAffine, ColorJitter, Normalize
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tqdm
from argparse import ArgumentParser
from configs.train_config import BATCH_SIZE, NUM_WORKERS, EPOCHS, DATA_DIR, IMG_SIZE, UNFREEZE_EPOCH
from dataset.dataset import Xray
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

from models.resnet import ResNet


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

def plot_confusion_matrix(writer,cm, class_name, epoch):
    pass
    figure = plt.figure(figsize=(20,20))
    plt.imshow(cm,interpolation="nearest", cmap="Wistia")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_name))
    plt.xticks(tick_marks, class_name, rotation=45)
    plt.yticks(tick_marks, class_name)

    cm = np.around(cm.astype('float')/cm.sum(axis=1)[:, np.newaxis], decimals=2)
    threshold = cm.max() / 2

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j,i, cm[i,j], horizontalalignment="center", color=color)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure("Confusion Matrix", figure, epoch)


def calculate_class_weights(dataset):
    labels = dataset.label_path
    class_counts = np.bincount(labels)
    total_samples = len(labels)

    class_weights = total_samples / (len(class_counts) * class_counts)

    # print(f"\nClass Distribution:")
    # for i, (count, weight) in enumerate(zip(class_counts, class_weights)):
    #     print(f"   Class {i} ({dataset.categories[i]}): {count} samples, weight: {weight:.4f}")

    return torch.FloatTensor(class_weights)


if __name__ == '__main__':

    args = get_args()

    if torch.cuda.is_available():
        print("Using GPU")
        device = torch.device('cuda')
    else:
        print("Using CPU")
        device = torch.device('cpu')

    train_transform = Compose([
        RandomAffine(
            degrees=(-5, 5),
            translate=(0.15, 0.15),
            shear=(-5, 5),
            scale=(0.8, 1.2),
        ),
        Resize((args.image_size, args.image_size)),
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    test_transform = Compose([
        Resize((args.image_size, args.image_size)),
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
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

    model = ResNet(freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.0001
    )

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        best_accuracy = checkpoint.get("best_accuracy", 0)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        start_epoch = 0
        best_accuracy = 0

    num_iters = len(train_loader)
    for epoch in range(start_epoch,args.epochs):
        if epoch == UNFREEZE_EPOCH:
            print("\nUnfreezing layer4")
            model.unfreeze_layer4()
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=1e-5
            )
        model.train()
        progress_bar = tqdm.tqdm(train_loader, colour='green')
        for iter, (images, labels) in enumerate(progress_bar):
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            output = model(images)
            loss = criterion(output, labels)
            progress_bar.set_description(
                f"Epoch [{epoch + 1}/{args.epochs}] | Iter [{iter + 1}/{num_iters}] | Loss: {loss.item():.3f}"
            )
            writter.add_scalar("Train/loss", loss, epoch*num_iters + iter)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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

            plot_confusion_matrix(writter, confusion_matrix(all_labels, all_preds),class_name=test_dataset.categories, epoch=epoch)
            accuracy = accuracy_score(all_labels, all_preds)
            print("Epoch {}: Accuracy {}".format(epoch + 1, accuracy))
            writter.add_scalar("Validation/accuracy", accuracy)
            checkpoint = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": loss,
            }
            torch.save(checkpoint, "{}/last_cnn.pt".format(args.model))
            # torch.save(model.state_dict(), "{}/last_cnn.pt".format(args.model))
            if best_accuracy < accuracy:
                checkpoint = {
                    "epoch": epoch + 1,
                    "best_accuracy": best_accuracy,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "loss": loss,
                }
                print("Best model saved")
                torch.save(checkpoint, "{}/best_resnet.pt".format(args.model))
                best_accuracy = accuracy
            # print(classification_report(all_labels, all_predictions))