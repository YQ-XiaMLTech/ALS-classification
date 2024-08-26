import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time

def train(model, train_loader, criterion, optimizer, device, classification):
    model.train()
    train_loss = 0.0
    train_acc = 0.0

    for image, label in train_loader:
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(image)
        if classification == 'binary-classification':
            label = label.view(-1, 1).float()
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * image.size(0)

        if classification == 'multi-classification':
            _, pred = torch.max(output, 1)
        elif classification == 'binary-classification':
            pred = (torch.sigmoid(output) > 0.5).float()

        correct_tensor = pred.eq(label.data.view_as(pred))
        accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
        train_acc += accuracy.item() * image.size(0)

    train_loss = train_loss / len(train_loader.dataset)
    train_acc = train_acc / len(train_loader.dataset)

    return train_loss, train_acc


# 定义验证函数
def validate(model, val_loader, criterion, device, classification):
    model.eval()
    val_loss = 0.0
    val_acc = 0.0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if classification == 'binary-classification':
                target = target.view(-1, 1).float()
            loss = criterion(output, target)

            val_loss += loss.item() * data.size(0)
            if classification == 'multi-classification':
                _, pred = torch.max(output, 1)
            elif classification == 'binary-classification':
                pred = (torch.sigmoid(output) > 0.5).float()

            correct_tensor = pred.eq(target.data.view_as(pred))
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            val_acc += accuracy.item() * data.size(0)

    val_loss = val_loss / len(val_loader.dataset)
    val_acc = val_acc / len(val_loader.dataset)

    return val_loss, val_acc

def train_attention(model, train_loader, criterion, optimizer, device, classification):
    model.train()
    train_loss = 0.0
    train_acc = 0.0

    for image, label in train_loader:
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        output, attention_weights_list = model(image)
        if classification == 'binary-classification':
            label = label.view(-1, 1).float()
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * image.size(0)

        if classification == 'multi-classification':
            _, pred = torch.max(output, 1)
        elif classification == 'binary-classification':
            pred = (torch.sigmoid(output) > 0.5).float()

        correct_tensor = pred.eq(label.data.view_as(pred))
        accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
        train_acc += accuracy.item() * image.size(0)

    train_loss = train_loss / len(train_loader.dataset)
    train_acc = train_acc / len(train_loader.dataset)

    return train_loss, train_acc


# 定义验证函数
def validate_attention(model, val_loader, criterion, device, classification):
    model.eval()
    val_loss = 0.0
    val_acc = 0.0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output, attention_weights_list = model(data)
            if classification == 'binary-classification':
                target = target.view(-1, 1).float()
            loss = criterion(output, target)

            val_loss += loss.item() * data.size(0)
            if classification == 'multi-classification':
                _, pred = torch.max(output, 1)
            elif classification == 'binary-classification':
                pred = (torch.sigmoid(output) > 0.5).float()

            correct_tensor = pred.eq(target.data.view_as(pred))
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            val_acc += accuracy.item() * data.size(0)

    val_loss = val_loss / len(val_loader.dataset)
    val_acc = val_acc / len(val_loader.dataset)

    return val_loss, val_acc, attention_weights_list