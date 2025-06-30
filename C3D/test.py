import pickle
import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import imageio
from torchsummary import summary
import cv2
from C3D_model import C3D
import random

def get_1x_lr_params(model):
    """
    This generator returns all the parameters for conv and two fc layers of the net.
    """
    b = [model.conv1, model.conv2, model.conv3a, model.conv3b, model.conv4a, model.conv4b,
         model.conv5a, model.conv5b, model.fc6, model.fc7]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    b = [model.fc8]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

nEpochs = 20  # Number of epochs for training
resume_epoch = 20  # Default is 0, change if want to resume
useTest = True  # See evolution of the test set when training
nTestInterval = 10  # Run on test set every nTestInterval epochs
snapshot = 10  # Store a model every snapshot epochs
lr = 1e-3  # Learning rate
batch_size = 1
num_worker = 0

dataset = 'ucf101'  # Options: hmdb51 or ucf101

if dataset == 'hmdb51':
    num_classes = 51
elif dataset == 'ucf101':
    num_classes = 101
else:
    print('We only implemented hmdb and ucf datasets.')
    raise NotImplementedError

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
# exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]


if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
modelName = 'C3D'  # Options: C3D or R2Plus1D or R3D
saveName = modelName + '-' + dataset


def test_model(dataset=dataset, save_dir=save_dir, num_classes=num_classes, lr=lr,
               num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval,
               batch_size=batch_size, num_worker=num_worker):
    if modelName == 'C3D':
        model = C3D(num_classes=num_classes, pretrained=True)
        train_params = [{'params': get_1x_lr_params(model), 'lr': lr},
                        {'params': get_10x_lr_params(model), 'lr': lr * 10}]
    else:
        print('We only implemented C3D models.')
        raise NotImplementedError

    criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                          gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs

    if resume_epoch == 0:
        print("Training {} from scratch...".format(modelName))
    else:
        checkpoint = torch.load(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar'),
            map_location=lambda storage, loc: storage)  # Load all tensors onto the CPU
        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)
    criterion.to(device)

    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    # writer = SummaryWriter(log_dir=log_dir)

    test_data = VideoDataset(dataset=dataset, split='test', clip_len=16)
    test_dataloader = DataLoader(test_data, batch_size=batch_size,
                                 num_workers=num_worker)

    test_size = len(test_dataloader.dataset)
    true_label = [0 for i in range(101)]
    predict_False_label = [0 for i in range(101)]
    if useTest:
        model.eval()
        start_time = timeit.default_timer()

        running_loss = 0.0
        running_corrects = 0.0
        result = []

        for i, (inputs, labels) in tqdm(enumerate(test_dataloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
            probs = nn.Softmax(dim=1)(outputs)
            preds = torch.max(probs, 1)[1]

            videodict = test_data.getlabel2index()
            labels = labels.to(torch.int64)
            loss = criterion(outputs, labels)
            for j in range(len(preds)):
                result.append(
                    [str(preds[j].cpu().numpy()), str(labels[j].cpu().numpy()), test_data.fnames[i * batch_size + j]])

            running_loss += loss.item() * inputs.size(0)
            true_label[int(labels)] += 1
            if preds != labels:
                predict_False_label[int(labels)] += 1

            running_corrects += torch.sum(preds == labels)

        epoch_loss = running_loss / test_size
        epoch_acc = running_corrects.double() / test_size
        result = np.array(result)
        # np.savetxt("result.csv", result,delimiter=',')
        with open('result.pk', 'wb') as file:
            pickle.dump(result, file)

        # writer.add_scalar('data/test_loss_epoch', epoch_loss, epoch)
        # writer.add_scalar('data/test_acc_epoch', epoch_acc, epoch)

        print("[test] Loss: {} Acc: {}".format(epoch_loss, epoch_acc))
        stop_time = timeit.default_timer()
        print("Execution time: " + str(stop_time - start_time) + "\n")

    ziped = list(zip(list(range(101)), predict_False_label))
    tmp = []
    for i in range(101):
        tmp.append([i, ziped[i][1] / true_label[i]])
    ziped = tmp
    ziped = sorted(ziped, key=lambda a: a[1])
    class_dir = os.path.join('data', 'UCF-101')
    sorted_classes = os.listdir(class_dir)
    for i in range(101):
        print(sorted_classes[i], ziped[i])


def crop(buffer, crop_size=112):
    # Randomly select start indices in order to crop the video
    height_index = np.random.randint(buffer.shape[0] - crop_size)
    width_index = np.random.randint(buffer.shape[1] - crop_size)

    buffer = buffer[
             height_index:height_index + crop_size,
             width_index:width_index + crop_size, :]

    return buffer


def plot_image():
    """
    plot the git image.
    :return:
    """
    class_dir = os.path.join('data', 'UCF-101')
    sorted_classes = os.listdir(class_dir)
    label_name = sorted_classes[random.randint(0, len(sorted_classes) - 1)]
    video_dir = os.path.join(class_dir, label_name)
    videos = os.listdir(video_dir)
    video_name = videos[random.randint(0, len(videos) - 1)]
    video = os.path.join(video_dir, video_name)

    model = C3D(num_classes=101, pretrained=True)
    checkpoint = torch.load(
        'run' + os.path.sep + 'run_0' + os.path.sep + 'models' + os.path.sep + 'C3D-ucf101_epoch-19.pth.tar',
        map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    #     summary(model, input_size= ( 3, 16, 112, 112))

    cap = cv2.VideoCapture(video)
    retaining = True
    clip = []
    frames = []

    while finish:
        finish, frame = cap.read()
        if not retaining and frame is None:
            continue

        croped_image = cv2.resize(frame, (112, 112))
        normalized_image = croped_image - np.array([[[90.0, 98.0, 102.0]]])
        clip.append(normalized_image)
        if len(clip) == 16:
            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            inputs = torch.from_numpy(inputs)
            inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)
            with torch.no_grad():
                print(inputs.shape)
                outputs = model.forward(inputs)

            probs = torch.nn.Softmax(dim=1)(outputs)
            label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]

            cv2.putText(frame, sorted_classes[label].split(' ')[-1].strip(), (25, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1)

            cv2.putText(frame, "prob: %.4f" % probs[0][label], (25, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1)
            clip.pop(0)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cv2.imshow('predicted', frame)
        cv2.waitKey(60)

    imageio.mimsave(label_name + '.gif', np.array(frames))
    cap.release()
    cv2.destroyAllWindows()


def crop(buffer, crop_size=112):
    height_index = int((buffer.shape[0] - crop_size) / 2)
    width_index = int((buffer.shape[1] - crop_size) / 2)
    buffer = buffer[height_index:height_index + crop_size, width_index:width_index + crop_size:]
    return buffer


def plot_image1():
    """
    this plot the large image with 81 examples.
    :return:
    """

    class_dir = 'data/ucf101/test'
    # folder = os.path.join(self.output_dir, split)
    classes = os.listdir(class_dir)[:81]
    images = []
    for class_name in classes:
        pre_path = os.path.join(class_dir, class_name)
        image_path = os.path.join(os.path.join(pre_path, os.listdir(pre_path)[0]), '00007.jpg')
        t = cv2.imread(image_path)
        crop_t = crop(t)
        cv2.putText(crop_t, class_name, (1, 100),
                    cv2.FONT_ITALIC, 0.5,
                    (128, 255, 255), 1)
        images.append(crop_t)

    stacked_image = np.hstack(images[0:  9])
    for i in range(1, 9):
        stacked_image = np.vstack((stacked_image, np.hstack(images[i * 9: i * 9 + 9])))

    print(stacked_image.shape)
    cv2.imshow('image', stacked_image)
    cv2.imwrite('91pics.jpg', stacked_image)
    cv2.waitKey()


if __name__ == '__main__':
    test_model()
    plot_image()
    # plot_image1()

