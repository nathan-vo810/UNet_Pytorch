import sys
import os
import numpy as np
from optparse import OptionParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from model import UNet


def train_net(net,
              epochs=5,
              batch_size=1,
              lr=1e-3,
              val_percent=0.04,
              save_cp=True, ):
    dir_img = 'data/train/'
    dir_label = 'data/label/'
    dir_checkpoint = 'checkpoints/'

    print('''Starting training:
    Epochs: {}
    Batch size: {}
    Learning rate: {}
    '''.format(epochs, batch_size, lr))

    optimizer = optim.Adam(net.parameters(),
                           lr=lr,
                           betas=(0.9, 0.999),
                           weight_decay=5e-4)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))

        epoch_loss = 0

        for i, b in enumerate(batch(train, batch_size)):
            imgs = np.array([i[0] for i in b]).astype(np.float32)
            labels = np.array([i[1] for i in b])

            imgs = torch.from_numpy(imgs)
            labels = torch.from_numpy(labels)

            labels_predict = net(imgs)
            labels_probs = F.sigmoid(labels_predict)
            labels_probs_flat = labels_probs.view(-1)

            labels_flat = labels.view(-1)

            loss = criterion(labels_probs_flat, labels_flat)
            epoch_loss += loss.item()

            print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch finished! Loss: {}'.format(epoch_loss / i))

        if save_cp:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved!'.format(epoch + 1))


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=1,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()

    net = UNet(n_channels=3, n_classes=1)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
