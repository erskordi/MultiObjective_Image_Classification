import numpy as np
import torch
from torchsummary import summary
from torch import nn
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from vit import ViT
from dataloader import image_data_import

def ovr_confusion_matrices_standard(cm):
    cm = np.asarray(cm)
    total = cm.sum()
    K = cm.shape[0]
    out = []
    tpr, fpr = [], []

    for k in range(K):
        tp = cm[k, k]
        fn = cm[k, :].sum() - tp
        fp = cm[:, k].sum() - tp
        tn = total - tp - fn - fp

        ovr_cm = np.array([[tn, fp],
                           [fn, tp]])
        out.append(ovr_cm)
        tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)

    return out, tpr, fpr

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train() # Put the model in training mode.
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        # Place the model and predictions on the same device.
        X, y = X.to(device), y.to(device)
        
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        # This step updates the parameter values inside your network.
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Print progress every 100 batches:
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            total_loss += loss
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return total_loss

def validation(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() # Put the model in testing mode.
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 100*correct, test_loss

def test(dataloader, model, classes, device, monitor):
    # Generate confusion matrix and accuracy metrics for the test dataset.
    size = len(dataloader.dataset)
    model.eval() # Put the model in testing mode.
    correct = 0
    cm = [[0 for _ in range(len(classes))] for _ in range(len(classes))]
    monitor.begin_window("Test Evaluation")
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            for t, p in zip(y.view(-1), pred.argmax(1).view(-1)):
                cm[t.long()][p.long()] += 1
    correct /= size
    
    measurement = monitor.end_window("Test Evaluation")
    print(f"Energy consumed during test evaluation: {round(measurement.total_energy / measurement.time, 2)} W")
    return cm, correct

def train_model(train_dataloader, valid_dataloader, device, monitor):
    vit_model = ViT(
        img_size=64, 
        patch_size=16, 
        in_channels=1, 
        num_classes=1,#len(classes), 
        embed_dim=768,
        depth=12, 
        num_heads=12, 
        ff_dim=3072, 
        dropout=0.1
    ).to(device)
    #parameter_report(vit_model)
    summary(vit_model, (1, 64, 64))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(vit_model.parameters(), lr=1e-3)

    epochs = 10 # Try 5, 10 but not too many!
    training_loss, accuracy, test_loss = [], [], []
    monitor.begin_window("training")
    for t in range(epochs):
        monitor.begin_window("epoch")
        print(f"Epoch {t+1}\n-------------------------------")
        loss = train(train_dataloader, vit_model, loss_fn, optimizer, device=device)
        acc, avg_loss = validation(valid_dataloader, vit_model, loss_fn, device=device)
        training_loss.append(loss)
        accuracy.append(acc)
        test_loss.append(avg_loss)
        measurement = monitor.end_window("epoch")
        print(f"Epoch {t}: {round(measurement.time,2)} s, {round(measurement.total_energy,2)} J")
    print("Done!")
    measurement = monitor.end_window("training")
    print(f"Entire training: {round(measurement.time,2)} s, {round(measurement.total_energy,2)} J")

def test_model(test_dataloader, vit_model, device, classes, monitor):
    cm, acc = test(test_dataloader, vit_model, classes, device, monitor)
    print(f"Test Accuracy: {acc:.2f}%")
    print("Confusion Matrix:", np.array(cm))

    plt.figure(figsize=(10, 10)) # Optional: Adjust figure size
    sns.heatmap(np.array(cm), annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Confusion Matrix")
    plt.show()

    out, tpr, fpr = ovr_confusion_matrices_standard(cm)

    auc = 0
    for i, (ovr_cm, tpr_val, fpr_val) in enumerate(zip(out, tpr, fpr)):
        auc += (1 + tpr_val - fpr_val) / 2
    print(f"Average AUC across classes: {auc / len(classes):.4f}")