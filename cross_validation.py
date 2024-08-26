import torch.optim as optim
import copy
from sklearn.model_selection import KFold
import time
import torch
import torch.nn as nn
import os
import os.path as osp
import cv2
import argparse
import numpy as np
import shutil
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from process_dataset import split_data
from datetime import datetime
from config import config
from process_dataset import pre_data
from process_dataset import utils_image
from process_dataset.split_data import ImageClassifyDataset
from torchvision import transforms
from model.CNNModel import Multi_CNNModel,Bin_CNNModel
from model.DenseNet import FineTunedResNet
from model import utils
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, cohen_kappa_score, accuracy_score
import seaborn as sns

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, patience, path_to_save):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0

    # 初始化记录训练和验证损失/准确率的列表
    epoch_train_loss_list = []
    epoch_train_acc_list = []
    epoch_val_loss_list = []
    epoch_val_acc_list = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 每个epoch都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()  # 设置模型为评估模式

            running_loss = 0.0
            running_corrects = 0

            # 迭代数据
            for inputs, labels in (train_loader if phase == 'train' else val_loader):
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)

                # 清零参数梯度
                optimizer.zero_grad()

                # 前向传播
                # 训练时跟踪历史记录
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 后向传播 + 仅在训练阶段进行优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # 计算损失和准确率
            epoch_loss = running_loss / len(train_loader.dataset if phase == 'train' else val_loader.dataset)
            epoch_acc = running_corrects.double() / len(train_loader.dataset if phase == 'train' else val_loader.dataset)

            # 记录训练和验证损失/准确率
            if phase == 'train':
                epoch_train_loss_list.append(epoch_loss)
                epoch_train_acc_list.append(epoch_acc)
            else:
                epoch_val_loss_list.append(epoch_loss)
                epoch_val_acc_list.append(epoch_acc)
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                scheduler.step(epoch_loss)  # 在验证阶段后更新学习率

            # 深拷贝模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), path_to_save)
                epochs_no_improve = 0
            elif phase == 'val':
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print('Early stopping')
                    model.load_state_dict(best_model_wts)
                    # 返回模型和记录的统计数据
                    return model, epoch_train_loss_list, epoch_train_acc_list, epoch_val_loss_list, epoch_val_acc_list

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model, epoch_train_loss_list, epoch_train_acc_list, epoch_val_loss_list, epoch_val_acc_list


if __name__ == "__main__":
    # def process():
    #     generate_dataset(config.process_src_path, config.process_dst_path, config)
    parser = argparse.ArgumentParser()
    # parser.add_argument("--epoch", type=int, default=config.epoch, help="epoch")
    parser.add_argument("--batch_size", type=int, default=config.batch_size, help="batch_size")
    parser.add_argument('--lr', type=float, default=config.lr, help='learning rate, default=0.01')
    parser.add_argument("--classification", type=str, default=config.classification, help="classification")
    parser.add_argument("--type_classification", type=str, default=config.type_classification, help="type_classification")
    parser.add_argument("--dataset", type=str, default=config.dataset, help="dataset")
    parser.add_argument("--label", type=str, default=config.label, help="label")
    # parser.add_argument("--dataset_save_as", type=str, default=config.dataset+config.model, help="dataset_save_as")
    # parser.add_argument("--train_ratio", type=int, default=config.train_ratio, help="important: train_ratio")
    # parser.add_argument("--test_ratio", type=int, default=config.test_ratio, help="important: test_ratio")
    # parser.add_argument("--val_ratio", type=int, default=config.val_ratio, help="important: val_ratio")
    parser.add_argument("--gpu", type=int, default=True, help="using gpu or not")
    parser.add_argument("--model", type=str, default= config.model, help="model")

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")  # device = "cpu"


    config.timestring = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    model_config_print = config.__dict__.copy()
    print("Loaded full config successfully:")
    # print("config =", json.dumps(model_config_print, indent=4))
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[Step 1] Configurations")
    print("using: {}".format(config.device))
    for item in config.__dict__.items():
        if item[0][0] == "_":
            continue
        print("{}: {}".format(item[0], item[1]))

    print("[Step 3] Initializing model")
    batch = 'batch_size' + str(args.batch_size)
    lr = 'lr' + str(args.lr)
    type_classification = str(args.classification) + '_' + str(args.type_classification)
    path = ('-').join([config.timestring, type_classification, batch, lr])
    main_save_path = osp.join(config.main_path, "saves", path)
    if not os.path.exists(main_save_path):
        os.makedirs(main_save_path)

    model_save_path = osp.join(main_save_path, "model_last.pt")
    figure_save_path_loss_and_acc = osp.join(main_save_path, "loss_and_acc.png")
    figure_save_path_accuracy_per_trial = osp.join(main_save_path, "accuracy_per_trial.png")
    print("main_save_path: {}".format(main_save_path))
    print("model_save_path: {}".format(model_save_path))
    # print("figure_save_path_confusion_matrix: {}".format(figure_save_path_confusion_matrix))
    print("figure_save_path_loss_whole: {}".format(figure_save_path_loss_and_acc))
    print("accuracy_per_trial: {}".format(figure_save_path_accuracy_per_trial))



    trial_params = []
    val_acc_history = []
    # 随机搜索超参数
    num_random_trials = 1  # 随机搜索的次数
    # 在随机搜索循环中
    best_trial_acc = 0

    for trial in range(num_random_trials):

        if args.model == "CNN":
            # 随机选择超参数
            lr = 10 ** np.random.uniform(-7, -3)
            batch_size = int(np.random.choice([32, 64, 128, 256]))

            dropout_rate1 = np.random.uniform(0.0, 0.5)
            dropout_rate2 = np.random.uniform(0.0, 0.5)
            dropout_rate3 = np.random.uniform(0.0, 0.5)
            dropout_rate4 = np.random.uniform(0.0, 0.5)
            dropout_rate5 = np.random.uniform(0.0, 0.5)
            scheduler_patience_options = int(np.random.choice([2, 5, 10]))

            print(
                f"Trial {trial}: lr={lr}, batch_size={batch_size}, dropout1={dropout_rate1},dropout2={dropout_rate2},dropout3={dropout_rate3},dropout4={dropout_rate4},dropout5={dropout_rate5},scheduler_patience_options={scheduler_patience_options}")
            trial_params.append({
                'lr': lr,
                'batch_size': batch_size,
                'dropout_rate1': dropout_rate1,
                'dropout_rate2': dropout_rate2,
                'dropout_rate3': dropout_rate3,
                'dropout_rate4': dropout_rate4,
                'dropout_rate5': dropout_rate5,
                'scheduler_patience_options': scheduler_patience_options,
            })
        elif args.model == "ResNet":
            lr = 10 ** np.random.uniform(-5, -1)
            dropout_rate = np.random.uniform(0.0, 0.5)
            weight_decay_options = np.random.choice(np.logspace(-5, -2, base=10))
            batch_size = int(np.random.choice([16, 32, 64, 128]))
            scheduler_patience_options = int(np.random.choice([2, 5, 10]))
            scheduler_factor_options = int(np.random.choice([0.1, 0.25, 0.5]))
            unfreeze_layers = int(np.random.choice(range(1, 9)))

            print(
                f"Trial {trial}: lr={lr}, batch_size={batch_size}, dropout={dropout_rate},weight_decay_options={weight_decay_options},"
                f"scheduler_patience_options={scheduler_patience_options},scheduler_factor_options={scheduler_factor_options}，unfreeze_layers={unfreeze_layers}")
            trial_params.append({
                'lr': lr,
                'batch_size': batch_size,
                'dropout_rate': dropout_rate,
                'weight_decay_options': weight_decay_options,
                'scheduler_patience_options': scheduler_patience_options,
                'scheduler_factor_options': scheduler_factor_options,
                'unfreeze_layers':unfreeze_layers
            })

        print("[Step 2] Preparing dataset...")
        dataset_path = args.dataset
        label_path = args.label
        images, labels = pre_data.read_files(label_path, args.classification, args.type_classification)
        # train_imagedir, train_labels, val_imagedir,val_labels,test_imagedir,test_labels = pre_data.split_datasets(dataset_path,images,labels, args.train_ratio, args.val_ratio, args.test_ratio)
        # train_imagedir, train_labels, test_imagedir, test_labels = pre_data.split_datasets_notvalortest(dataset_path, images, labels, args.train_ratio, args.test_ratio)
        # train_imagedir, train_labels, val_imagedir, val_labels = pre_data.split_datasets_notvalortest(dataset_path, images, labels, args.train_ratio, args.val_ratio)
        # 将字典转换为列表，保持图像和标签之间的对应关系
        image_paths = [f"{dataset_path}/{img}" for img in images.values()]
        image_labels = [lbl for lbl in labels.values()]
        # transform
        mean, std = pre_data.compute_mean_std(dataset_path)
        # mean, std = pre_data.gray_compute_mean_std(dataset_path)
        transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.Resize((400, 400)),
                                        transforms.ToTensor(), transforms.Normalize(mean=mean, std=std), ])
        # dataset
        dataset = ImageClassifyDataset(image_paths, image_labels, transform=transform)
        # 初始化模型、优化器、损失函数
        if args.model == "ResNet":
            # Create an instance of the FineTunedResNet model
            model = FineTunedResNet(num_classes=3, dropout_rate=dropout_rate, unfreeze_layers=unfreeze_layers).to(config.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                         lr=lr,
                                         weight_decay=weight_decay_options)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor_options,
                                          patience=scheduler_patience_options, min_lr=0.00001)
            # shashasha
        elif args.model == "CNN":
            model = Multi_CNNModel(dropout_rate1, dropout_rate2, dropout_rate3, dropout_rate4, dropout_rate5).to(
                args.device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=scheduler_patience_options, min_lr=0.00001)

        print(model)

        k_folds = 4
        kfold = KFold(n_splits=k_folds, shuffle=True)
        fold_performance = []

        for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
            print(f"FOLD {fold}")
            print("---------------------------------------------------")

            # Prepare data loaders for this fold
            train_subset = Subset(dataset, train_ids)
            val_subset = Subset(dataset,val_ids)
            # 获取训练集的图像路径和标签
            train_image_paths = [dataset.image_paths[idx] for idx in train_subset.indices]
            train_labels = [dataset.labels[idx] for idx in train_subset.indices]
            val_image_paths = [dataset.image_paths[idx] for idx in val_subset.indices]
            val_labels = [dataset.labels[idx] for idx in val_subset.indices]

            # data augmentation，只对train数据集做变换，先进行直方图均衡化，然后根据方差判断如果方差小于1300，提高对比度，最后边缘增强
            print("[Step 3] data augmentation")

            # 图像处理代码开始
            output_folder = 'dataset/data_augmentation'
            if os.path.exists(output_folder):
                # 删除文件夹中的所有文件和子文件夹
                for filename in os.listdir(output_folder):
                    file_path = os.path.join(output_folder, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f'Failed to delete {file_path}. Reason: {e}')
            else:
                # 如果文件夹不存在，则创建它
                os.makedirs(output_folder)
            # 对训练数据进行增强
            augmented_image_paths, augmented_labels = utils_image.augment_images(train_image_paths, train_labels,
                                                                                 output_folder)
            assert len(augmented_image_paths) == len(
                augmented_image_paths), "The lengths of train_imagedir and train_labels are not equal!"

            print('train_imagedir', train_image_paths, '\ntrain_labels', train_labels)
            print('val_imagedir', val_image_paths, '\nval_labels', val_labels)
            # print('test_imagedir',test_imagedir, '\ntest_labels',test_labels)
            # print('train:', len(train_imagedir), '\ntest:', len(test_imagedir))
            print('train:', len(train_image_paths), '\nval:', len(val_image_paths))
            print('augmented_image_paths', augmented_image_paths, '\naugmented_labels', augmented_labels)
            print('augmented_image_len', len(augmented_image_paths), '\naugmented_labels_len', len(augmented_labels))
            # print('train:',len(train_imagedir),'\nval:',len(val_imagedir),'\ntest:',len(test_imagedir))
            print(f"Number of 0s: {train_labels.count(0)}")
            print(f"Number of 1s: {train_labels.count(1)}")
            print(f"Number of 2s: {train_labels.count(2)}")



            # 使用增强后的图像和标签创建新的训练数据集
            augmented_train_dataset = ImageClassifyDataset(augmented_image_paths, augmented_labels, transform=transform)

            # 创建DataLoader
            train_loader = DataLoader(augmented_train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)


            # 调用训练函数
            model_save_path = os.path.join(main_save_path, f"model_trial_{trial}.pt")
            model, train_loss_list, train_acc_list, val_loss_list, val_acc_list = train_model(
                model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs=1000, patience=scheduler_patience_options + 2, path_to_save=model_save_path
            )

            fold_performance.append({
                'fold': fold,
                'val_accuracy': val_acc_list[-1]
            })

        # 打印每一折的性能
        for performance in fold_performance:
            print(
                f"Fold {performance['fold']}: Accuracy = {performance['val_accuracy']}")

        # 计算所有折的平均性能
        average_accuracy = sum(d['val_accuracy'] for d in fold_performance) / k_folds
        print(f"Average Accuracy: {average_accuracy}")
        val_acc_history.append(average_accuracy)

    # 计算并记录最佳验证集准确率
    trial_best_val_acc = max(val_acc_history)
    best_trial_idx = val_acc_history.index(trial_best_val_acc)
    best_params = trial_params[best_trial_idx]

    print(f"The best trial is trial {best_trial_idx} with parameters:")
    if args.model == "CNN":
        print(f"Learning Rate: {best_params['lr']}")
        print(f"Batch Size: {best_params['batch_size']}")
        print(f"Dropout Rate1: {best_params['dropout_rate1']}")
        print(f"Dropout Rate2: {best_params['dropout_rate2']}")
        print(f"Dropout Rate3: {best_params['dropout_rate3']}")
        print(f"Dropout Rate4: {best_params['dropout_rate4']}")
        print(f"Dropout Rate5: {best_params['dropout_rate5']}")
    elif args.model == "ResNet":
        print(f"Learning Rate: {best_params['lr']}")
        print(f"Batch Size: {best_params['batch_size']}")
        print(f"dropout_rate: {best_params['dropout_rate']}")
        print(f"weight_decay_options: {best_params['weight_decay_options']}")
        print(f"scheduler_patience_options: {best_params['scheduler_patience_options']}")
        print(f"scheduler_factor_options: {best_params['scheduler_factor_options']}")


    val_acc_history_np = [v.cpu().numpy() for v in val_acc_history]
    # 绘制所有尝试的最佳验证准确率折线图
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_random_trials + 1), val_acc_history_np, marker='o', linestyle='--')
    plt.title('Validation Accuracy per Trial')
    plt.xlabel('Trial')
    plt.ylabel('Validation Accuracy')
    plt.grid(True)
    if figure_save_path_accuracy_per_trial is not None:
        plt.savefig(figure_save_path_accuracy_per_trial)
        plt.close()
    else:
        plt.show()












