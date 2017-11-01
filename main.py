import os
import datetime
import random
import time
import json
import tqdm

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch import optim
from torch.autograd import Variable
#from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
#import torchvision.models as models

from tensorboardX import SummaryWriter

from datasets import scenedata
from models.resnet import *
from utils import adjust_learning_rate, save_checkpoint, AverageMeter, accuracy

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
num_classes = 80
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

#writer = SummaryWriter('logs')

args = {
	'resume': 'model_best.pth.tar',
	'batch_size': 20,
	'weight_decay': 1e-4,
	'momentum': 0.9,
	'print_freq': 200,
	'epoch_num': 60,
	'lr': 0.01,
	'model': 'resnet152'
}


def main():

    start_epoch = 0
    best_prec1 = 0
    
    model = ResNet(pretrained=True, num_classes=num_classes).cuda()

    if args['resume']:
        if os.path.isfile(args['resume']):
            print("=> loading checkpoint '{}'".format(args['resume']))
            checkpoint = torch.load(args['resume'])
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'(epoch {})"
                .format(args['resume'], checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args['resume']))


    cudnn.benchmark = True

    # data loader
    train_transforms = transforms.Compose([
        transforms.RandomSizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*mean_std),])

    val_transforms = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(*mean_std),])

    train_set = scenedata.SceneDataset('train', transform=train_transforms)
    train_loader = DataLoader(train_set, batch_size=args['batch_size'], num_workers=4, shuffle=True)
    val_set = scenedata.SceneDataset('val', transform=val_transforms)
    val_loader = DataLoader(val_set, batch_size=args['batch_size'], num_workers=4, shuffle=False)


    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()

    optimizer = optim.SGD(model.parameters(), args['lr'], momentum=args['momentum'],
            weight_decay=args['weight_decay'])


    for epoch in range(start_epoch, args['epoch_num']):
        adjust_learning_rate(optimizer, epoch, args['lr'])
            
        train(train_loader, model, criterion, optimizer, epoch)
            
        prec1 = validate(val_loader, model, criterion, epoch)
            
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
	    'model': args['model'],
	    'state_dict': model.state_dict(),
	    'best_prec1': best_prec1,
	    }, is_best, args['model'])



def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    curr_iter = epoch * len(train_loader)

    model.train()

    for i, data in enumerate(train_loader):
        
        inputs, labels = data
        N = inputs.size(0)
        
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        prec1, prec3 = accuracy(outputs.data, labels.data, topk=(1,3))
        losses.update(loss.data[0], N)
        top1.update(prec1[0], N)
        top3.update(prec3[0], N)
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

        curr_iter += 1
        writer.add_scalar('train_loss', losses.avg, curr_iter)
        
        
        if i % args['print_freq'] == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                   epoch, i, len(train_loader),loss=losses, top1=top1, top3=top3))



def validate(val_loader, model, criterion, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    model.eval()

    for i, data in enumerate(val_loader):
        inputs, labels = data
        
        N = inputs.size(0)
        
        inputs = Variable(inputs, volatile=True).cuda()
        labels = Variable(labels, volatile=True).cuda()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        prec1, prec3 = accuracy(outputs.data, labels.data, topk=(1,3))
        losses.update(loss.data[0], N)
        top1.update(prec1[0], N)
        top3.update(prec3[0], N)
         
        if i % args['print_freq'] == 0:
            print('Test: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                   i, len(val_loader), loss=losses, top1=top1, top3=top3))

    print(' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'
          .format(top1=top1, top3=top3))

    writer.add_scalar('val_loss', losses.avg, epoch)
    writer.add_scalar('top1_acc', top1.avg, epoch)
    writer.add_scalar('top3_acc', top3.avg, epoch)

    return top1.avg


def submit():

    model = ResNet(pretrained=False, num_classes=num_classes)

    model.load_state_dict(torch.load('model_best.pth.tar')['state_dict'])

    model.eval().cuda()

    val_transforms = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(*mean_std),])


    test_set = scenedata.SceneDataset('test', transform=val_transforms)
    test_loader = DataLoader(test_set, batch_size=args['batch_size'], num_workers=4, 
            shuffle=False, pin_memory=True)

    results = []
    for ii, data in tqdm.tqdm(enumerate(test_loader)):
        inputs, img_ids = data
        inputs = Variable(inputs, volatile=True).cuda()
        score = model(inputs)
        preds = score.data.topk(k=3)[1].tolist()
        result = [ {"image_id": img_id,
                    "label_id": label_id} for img_id, label_id in zip(img_ids, preds)]

        results += result

    with open('baseline.json', 'w') as f:
        json.dump(results, f)



if __name__ == '__main__':
    #main()
    submit()










