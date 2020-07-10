import argparse
import os
import copy
from loss import HuberLoss, CharbonnierLoss
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from models import CSRCNN
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str,default= )
    parser.add_argument('--eval-file', type=str,default= )
    parser.add_argument('--eval-file1', type=str, default= )  
    parser.add_argument('--eval-file2', type=str, default= )
    parser.add_argument('--outputs-dir', type=str,default='BLAH_BLAH/outputs')
    parser.add_argument('--weights-file', type=str)
    parser.add_argument('--scale', type=int, default= )
    parser.add_argument('--lr', type=float, default= )
    parser.add_argument('--batch-size', type=int, default= )
    parser.add_argument('--num-epochs', type=int, default= )
    parser.add_argument('--num-workers', type=int, default= )
    parser.add_argument('--seed', type=int, default= )
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    model = CSRCNN(scale_factor=args.scale).to(device)
    criterion =CharbonnierLoss(delta=0.0001)#CharbonnierLoss(delta=0.0001)#HuberLoss(delta=0.9)#nn.L1Loss()# nn.MSELoss()
    optimizer = optim.Adam([
        {'params': model.first_part.parameters(), 'lr': args.lr * 0.1},
        # {'params': model.mid_part.parameters(), 'lr': args.lr * 0.1},
        {'params': model.last_part.parameters(), 'lr': args.lr * 0.1}
    ], lr=args.lr)
    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=False)#drop_last=False
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)
    eval_dataset1 = EvalDataset(args.eval_file1)
    eval_dataloader1 = DataLoader(dataset=eval_dataset1, batch_size=1)
    eval_dataset2 = EvalDataset(args.eval_file2)
    eval_dataloader2 = DataLoader(dataset=eval_dataset2, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0
    epoch_num=range(1,args.num_epochs+1)
    psrn=[]
    loss_num=[]
    psrn_Set14=[]
    psrn_BSD200=[]
    
    for epoch in range(args.num_epochs):
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * (0.1 ** (epoch // int(args.num_epochs * 0.8)))
            print(param_group['lr'])


        model.train()
        epoch_losses = AverageMeter()
        
        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size), ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_dataloader:
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)

                loss = criterion(preds, labels)

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))

        model.eval()
        epoch_psnr = AverageMeter()
        epoch_psnr1 = AverageMeter()
        epoch_psnr2 = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)  

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

        for data in eval_dataloader1:
            inputs1, labels1 = data

            inputs1 = inputs1.to(device)
            labels1 = labels1.to(device)

            with torch.no_grad():
                preds1 = model(inputs1).clamp(0.0, 1.0)  

            epoch_psnr1.update(calc_psnr(preds1, labels1), len(inputs1))

        for data in eval_dataloader2:
            inputs2, labels2 = data

            inputs2 = inputs2.to(device)
            labels2 = labels2.to(device)

            with torch.no_grad():
                preds2 = model(inputs2).clamp(0.0, 1.0)  

            epoch_psnr2.update(calc_psnr(preds2, labels2), len(inputs2))


        print('Set-5 eval psnr: {:.2f}'.format(epoch_psnr.avg))
        print('Set-14 eval psnr: {:.2f}'.format(epoch_psnr1.avg))
        print('BSD200 eval psnr: {:.2f}'.format(epoch_psnr2.avg))
 
        psrn.append(epoch_psnr.avg)
        psrn_Set14.append(epoch_psnr1.avg)
        psrn_BSD200.append(epoch_psnr2.avg)
        loss_num.append(epoch_losses.avg)

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())


    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))

