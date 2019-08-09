import numpy as np
import torch
from tensorboardX import SummaryWriter
# from torch import nn
from tqdm import tqdm

from config import device, print_freq, learning_rate, sigma, epochs
from data_gen import LJSpeechDataset
from models import WaveGlow, WaveGlowLoss
from utils import parse_args, save_checkpoint, AverageMeter, get_logger


def train_net(args):
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_loss = float('inf')
    writer = SummaryWriter()
    epochs_since_improvement = 0

    # Initialize / load checkpoint
    if checkpoint is None:
        # model
        model = WaveGlow()
        print(model)
        # model = nn.DataParallel(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    logger = get_logger()

    # Move to GPU, if available
    model = model.to(device)

    criterion = WaveGlowLoss(sigma)

    # Custom dataloaders
    train_dataset = LJSpeechDataset()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers)
    valid_dataset = LJSpeechDataset()
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.num_workers)

    # Epochs
    for epoch in range(start_epoch, epochs):
        # One epoch's training
        train_loss = train(train_loader=train_loader,
                           model=model,
                           optimizer=optimizer,
                           criterion=criterion,
                           epoch=epoch,
                           logger=logger)
        writer.add_scalar('Train_Loss', train_loss, epoch)

        lr = optimizer.lr
        print('\nLearning rate: {}'.format(lr))
        writer.add_scalar('Learning_Rate', lr, epoch)
        step_num = optimizer.step_num
        print('Step num: {}\n'.format(step_num))

        # One epoch's validation
        valid_loss = valid(valid_loader=valid_loader,
                           model=model,
                           criterion=criterion,
                           logger=logger)
        writer.add_scalar('Valid_Loss', valid_loss, epoch)

        # Check if there was an improvement
        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss, is_best)


def train(train_loader, model, optimizer, criterion, epoch, logger):
    model.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()

    # Batches
    for i, (batch) in enumerate(train_loader):
        # Move to GPU, if available
        mel, audio = batch
        mel = mel.to(device)
        audio = audio.to(device)

        # Forward prop.
        outputs = model((mel, audio))
        loss = criterion(outputs)

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())

        # Print status
        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader), loss=losses))

    return losses.avg


def valid(valid_loader, model, criterion, logger):
    model.eval()

    losses = AverageMeter()

    # Batches
    for data in tqdm(valid_loader):
        # Move to GPU, if available
        padded_input, padded_target, input_lengths = data
        padded_input = padded_input.to(device)
        padded_target = padded_target.to(device)
        input_lengths = input_lengths.to(device)

        with torch.no_grad():
            # Forward prop.
            pred, gold = model(padded_input, input_lengths, padded_target)
            loss, n_correct = cal_performance(pred, gold, smoothing=args.label_smoothing)

        # Keep track of metrics
        losses.update(loss.item())

    # Print status
    logger.info('\nValidation Loss {loss.val:.4f} ({loss.avg:.4f})\n'.format(loss=losses))

    return losses.avg


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
