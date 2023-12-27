import time
import torch
import torch.nn as nn
import os
import os.path as osp
import cv2
import argparse
import numpy as np
import shutil
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
from model.AttentionModel import SE_FineTunedEfficientNet, CBAM_FineTunedEfficientNet
from model.Network import FineTunedResNet,FineTunedEfficientNet
from model import utils
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, cohen_kappa_score, accuracy_score
import seaborn as sns


if __name__ == "__main__":
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    # def process():
    #     generate_dataset(config.process_src_path, config.process_dst_path, config)
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=config.epoch, help="epoch")
    parser.add_argument("--batch_size", type=int, default=config.batch_size, help="batch_size")
    parser.add_argument('--lr', type=float, default=config.lr, help='learning rate, default=0.01')
    parser.add_argument("--classification", type=str, default=config.classification, help="classification")
    parser.add_argument("--type_classification", type=str, default=config.type_classification, help="type_classification")
    parser.add_argument("--dataset", type=str, default=config.dataset, help="dataset")
    parser.add_argument("--label", type=str, default=config.label, help="label")
    # parser.add_argument("--dataset_save_as", type=str, default=config.dataset+config.model, help="dataset_save_as")
    parser.add_argument("--train_ratio", type=int, default=config.train_ratio, help="important: train_ratio")
    parser.add_argument("--test_ratio", type=int, default=config.test_ratio, help="important: test_ratio")
    parser.add_argument("--val_ratio", type=int, default=config.val_ratio, help="important: val_ratio")
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

    print("[Step 2] Preparing dataset...")
    dataset_path = args.dataset
    label_path = args.label
    images,labels = pre_data.read_files(label_path,args.classification, args.type_classification)
    train_imagedir, train_labels, val_imagedir,val_labels,test_imagedir,test_labels = pre_data.split_datasets(dataset_path,images,labels, args.train_ratio, args.val_ratio, args.test_ratio)
    # train_imagedir, train_labels, test_imagedir, test_labels = pre_data.split_datasets_notval(dataset_path, images, labels, args.train_ratio, args.test_ratio)

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

    flip_imagedir = utils_image.flip_images(train_imagedir,output_folder)
    update_labels = train_labels + train_labels
    rotate_imagedir = utils_image.rotate_images(train_imagedir,output_folder)
    update_labels = update_labels + train_labels
    brightness_imagedir = utils_image.enhance_brightness(train_imagedir, output_folder)
    update_labels = update_labels + train_labels
    gray_imagedir = utils_image.gray_images(train_imagedir, output_folder)
    update_labels = update_labels + train_labels
    gray_denoising_imagedir = utils_image.gray_denoising_images(train_imagedir, output_folder)
    update_labels = update_labels + train_labels
    color_denoising_imagedir = utils_image.color_denoising_images(train_imagedir, output_folder)
    update_labels = update_labels + train_labels
    enhanced_denoising_imagedir = utils_image.enhanced_denoising_images(train_imagedir, output_folder)
    update_labels = update_labels + train_labels
    scale_imagedir = utils_image.scale_images(train_imagedir, output_folder, scale_factor=0.9)
    update_labels = update_labels + train_labels
    crop_imagedir = utils_image.crop_images(train_imagedir, output_folder, crop_size=(10, 10))
    update_labels = update_labels + train_labels
    noise_imagedir = utils_image.add_noise_images(train_imagedir, output_folder, noise_level=10)
    update_labels = update_labels + train_labels
    # Applying perspective transformation to images
    perspective_imagedir = utils_image.perspective_transform_images(train_imagedir, output_folder)
    update_labels = update_labels + train_labels


    train_imagedir.extend(flip_imagedir)
    train_imagedir.extend(rotate_imagedir)
    train_imagedir.extend(brightness_imagedir)
    train_imagedir.extend(gray_imagedir)
    train_imagedir.extend(gray_denoising_imagedir)
    train_imagedir.extend(color_denoising_imagedir)
    train_imagedir.extend(enhanced_denoising_imagedir)
    train_imagedir.extend(scale_imagedir)
    train_imagedir.extend(crop_imagedir)
    train_imagedir.extend(noise_imagedir)
    train_imagedir.extend(perspective_imagedir)
    train_labels = update_labels
    assert len(train_imagedir) == len(train_labels), "The lengths of train_imagedir and train_labels are not equal!"

    print('train_imagedir',train_imagedir,'\ntrain_labels',train_labels)
    print('val_imagedir',val_imagedir,'\nval_labels',val_labels)
    print('test_imagedir',test_imagedir, '\ntest_labels',test_labels)
    # print('train:', len(train_imagedir), '\ntest:', len(test_imagedir))
    print('train:',len(train_imagedir),'\nval:',len(val_imagedir),'\ntest:',len(test_imagedir))
    print(f"Number of 0s: {train_labels.count(0)}")
    print(f"Number of 1s: {train_labels.count(1)}")
    print(f"Number of 2s: {train_labels.count(2)}")

    # transform
    mean, std = pre_data.compute_mean_std(dataset_path)
    # mean, std = pre_data.gray_compute_mean_std(dataset_path)
    if args.model == "EfficientNet":
        train_transform =  transforms.Compose([transforms.RandomHorizontalFlip(),transforms.Resize((240, 240)),
                                               transforms.ToTensor(),transforms.Normalize(mean=mean, std=std),])
        test_transform = transforms.Compose([transforms.Resize((240, 240)),transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    else:
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.Resize((400, 400)),
                                              transforms.ToTensor(), transforms.Normalize(mean=mean, std=std), ])
        test_transform = transforms.Compose(
            [transforms.Resize((400, 400)), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    # dataset
    train_dataset = ImageClassifyDataset(train_imagedir, train_labels, transform=train_transform)
    val_dataset = ImageClassifyDataset(val_imagedir, val_labels, transform=test_transform)
    test_dataset = ImageClassifyDataset(test_imagedir, test_labels, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print("[Step 3] Initializing model")
    batch = 'batch_size' + str(args.batch_size)
    lr = 'lr'+str(args.lr)
    type_classification =str(args.classification) +'_' + str(args.type_classification)
    path = ('-').join([config.timestring,type_classification,batch,lr])
    main_save_path = osp.join(config.main_path, "saves", path)
    if not os.path.exists(main_save_path):
        os.makedirs(main_save_path)

    model_save_path = osp.join(main_save_path, "model_last.pt")
    figure_save_path_loss_and_acc = osp.join(main_save_path, "loss_and_acc.png")
    figure_save_path_confusion_matrix = osp.join(main_save_path, "confusion_matrix.png")
    regression_result_test_true = f"{main_save_path}/test_true.npy"
    regression_result_test_pred = f"{main_save_path}/test_pred.npy"
    # figure_regression_train_path = f"{main_save_path}/regression_train.png"
    # figure_regression_val_path = f"{main_save_path}/regression_val.png"
    # figure_regression_test_path = f"{main_save_path}/regression_test.png"
    print("main_save_path: {}".format(main_save_path))
    print("model_save_path: {}".format(model_save_path))
    print("figure_save_path_confusion_matrix: {}".format(figure_save_path_confusion_matrix))
    print("figure_save_path_loss_whole: {}".format(figure_save_path_loss_and_acc))
    print("regression_result_test_true: {}".format(regression_result_test_true))
    print("regression_result_test_pred: {}".format(regression_result_test_pred))
    # print("figure_regression_train_path: {}".format(figure_regression_train_path))
    # print("figure_regression_val_path: {}".format(figure_regression_val_path))
    # print("figure_regression_test_path: {}".format(figure_regression_test_path))

    if args.classification == 'multi-classification':
        if args.model == "ResNet18":
            # Create an instance of the FineTunedResNet model
            model = FineTunedResNet(num_classes=3, dropout_rate=0.147).to(config.device)
            # Set up the loss function
            criterion = nn.CrossEntropyLoss()
            # Set up the optimizer, note that we only pass parameters that require gradients
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr, weight_decay=0.000295)
            # Set up the learning rate scheduler
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0, patience=5, min_lr=0.00001)
            # shashasha
        elif args.model == "CNN":
            model = Multi_CNNModel().to(config.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=5, min_lr=0.00001)
        elif args.model == "SE_EfficientNet":
            model = SE_FineTunedEfficientNet(num_classes=3, dropout_rate=0.5).to(config.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr,
                                         weight_decay=0.000295)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=5, min_lr=0.00001)
        elif args.model == "CBAM_EfficientNet":
            model = CBAM_FineTunedEfficientNet(num_classes=3, dropout_rate=0.5).to(config.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr,
                                         weight_decay=0.000295)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=5, min_lr=0.00001)


    elif args.classification == 'binary-classification':
        model = Bin_CNNModel().to(config.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=2, min_lr=0.00001)

    print(model)

    print("[Step 4] Training...")
    epoch_train_loss_list = []
    epoch_train_acc_list = []
    epoch_val_loss_list = []
    epoch_val_acc_list = []
    # early stopping
    best_val_loss = float("inf")
    best_model_state = None
    patience = 7  # 当验证集损失在连续5个epoch中没有改善时停止训练
    counter = 0  # 追踪验证损失没有改善的epoch数量

    for epoch in range(args.epoch):
        epoch_start_time = time.time()

        # Training
        train_loss, train_acc = utils.train(model, train_loader, criterion, optimizer, args.device, args.classification)
        epoch_train_loss_list.append(train_loss)
        epoch_train_acc_list.append(train_acc)

        # Validation
        val_loss, val_acc = utils.validate(model, val_loader, criterion, args.device, args.classification)
        epoch_val_loss_list.append(val_loss)
        epoch_val_acc_list.append(val_acc)
        scheduler.step(val_loss)

        # Logging
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
              (epoch + 1, args.epoch, time.time() - epoch_start_time, \
               train_acc, train_loss, val_acc, val_loss))
        # print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f' % \
        #       (epoch + 1, args.epoch, time.time() - epoch_start_time, \
        #        train_acc, train_loss,))

        # Check if early stopping condition is met
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            best_model_state = model.state_dict()  # 保存最佳模型的状态
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break  # Early stopping

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    epochs = range(1, len(epoch_train_loss_list) + 1)

    plt.figure(figsize=(20, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, epoch_train_loss_list, 'b', label='Training loss')
    plt.plot(epochs, epoch_val_loss_list, 'r', label='Validation loss')  # 添加验证损失的曲线
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, epoch_train_acc_list, 'b', label='Training acc')
    plt.plot(epochs, epoch_val_acc_list, 'r', label='Validation acc')  # 添加验证损失的曲线
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.legend()

    if figure_save_path_loss_and_acc is not None:
        plt.savefig(figure_save_path_loss_and_acc)
        plt.close()
    else:
        plt.show()

    # Test
    print("[Step 6] Testing...")
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    test_true_list = []
    test_pred_list = []

    total_error = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            if args.classification == 'multi-classification':
                _, pred = torch.max(output, 1)
            elif args.classification == 'binary-classification':
                pred = (torch.sigmoid(output) > 0.5).float()

            test_true_list.extend(target.cpu().numpy())
            test_pred_list.extend(pred.cpu().numpy())

    # 将列表转换为NumPy数组
    test_true_list = np.asarray(test_true_list)
    test_pred_list = np.asarray(test_pred_list)

    np.save(regression_result_test_true, test_true_list)
    np.save(regression_result_test_pred, test_pred_list)

    print("[Step 7] Drawing test result...")
    # 计算混淆矩阵
    cm = confusion_matrix(test_true_list, test_pred_list)
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    if figure_save_path_confusion_matrix is not None:
        plt.savefig(figure_save_path_confusion_matrix)
        plt.close()
    else:
        plt.show()

    # metrics for data imbalanced
    if args.classification == 'multi-classification':
        f1 = f1_score(test_true_list, test_pred_list, average='weighted')
        # mcc = matthews_corrcoef(test_true_list, test_pred_list)
        kappa = cohen_kappa_score(test_true_list, test_pred_list)
        accuracy = accuracy_score(test_true_list, test_pred_list)
        print("f1", f1, "\nkappa", kappa,"\naccuracy",accuracy)
    elif args.classification == 'binary-classification':
        f1 = f1_score(test_true_list, test_pred_list)
        mcc = matthews_corrcoef(test_true_list, test_pred_list)
        kappa = cohen_kappa_score(test_true_list, test_pred_list)
        accuracy = accuracy_score(test_true_list, test_pred_list)
        print("f1", f1, "\nmcc", mcc, "\nkappa", kappa,"\naccuracy",accuracy)




