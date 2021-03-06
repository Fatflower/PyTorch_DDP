# %Header File Start-----------------------------------------------------------
#  Confidential（Unclassified）
#  COPYRIGHT (C) Sun Yat-sen University
#  THIS FILE MAY NOT BE MODIFIED OR REDISTRIBUTED WITHOUT THE
#  EXPRESSED WRITTEN CONSENT OF SYSU
# 
# %-----------------------------------------------------------------------------
#  Title   : DDP_test_data.py
#  Author  : Zhang wentao; 
#  E-mail  : z1282429194@163.com
#  Created : 10/14/2021
#  Description: An example of DistributedDataParallel training under the pytorch
#               framework. This example contains how to load data and models, 
#               and save the model. The main reference for this: https://zhuanlan.zhihu.com/p/419833524
# %-----------------------------------------------------------------------------
#  Modification History:
#  V1.0: 2021.10.14, first created by Zhang wentao
#
# %Header File End--------------------------------------------------------------


import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Resnet_refine import ResNet_refine

from torch.nn.parallel import DistributedDataParallel as DDP
from apex import amp # install apex reference: https://blog.csdn.net/Orientliu96/article/details/104583998
from prefetch_generator import BackgroundGenerator # pip install prefetch_generator


class DataLoaderX(DataLoader):
    """(加速组件) 重新封装Dataloader，使prefetch不用等待整个iteration完成"""
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def reduce_tensor(tensor, world_size):
    # 用于平均所有gpu上的运行结果，比如loss
    # Reduces the tensor data across all machines
    # Example: If we print the tensor, we can get:
    # tensor(334.4330, device='cuda:1') *********************, here is cuda:  cuda:1
    # tensor(359.1895, device='cuda:3') *********************, here is cuda:  cuda:3
    # tensor(263.3543, device='cuda:2') *********************, here is cuda:  cuda:2
    # tensor(340.1970, device='cuda:0') *********************, here is cuda:  cuda:0
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.110'
    os.environ['MASTER_PORT'] = '30000'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run(rank, world_size):
    setup(rank, world_size)

    torch.manual_seed(18)
    torch.cuda.manual_seed_all(18)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(rank) # 这里设置 device ，后面可以直接使用 data.cuda(),否则需要指定 rank



    # load model
    model = ResNet_refine('resnet18', False, 10).to(rank)
    # Replace the BN in the model
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    # amp.initialize 将模型和优化器为了进行后续混合精度训练而进行封装。注意，在调用 amp.initialize 之前，模型模型必须已经部署在GPU上。
    model, optimizer = amp.initialize(model, optimizer, opt_level='O1') # 这里是“欧一”，不是“零一”

    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True) #  
    
    start_epoch = 0
    # Determine whether to load checkpoint
    # resume = 0 : load checkpoint
    # resume = 1 : don't load checkpoint
    resume = 0
    if resume == 0:
        print('resuming from checkpoint...')
        #reference :https://github.com/pytorch/pytorch/issues/23138
        checkpoint = torch.load('./checkpoint/resnet_cifar10_checkpoint.pth', map_location='cuda:{}'.format(rank)) 
        model.load_state_dict(checkpoint['model'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    batch_size = 256
    train_dataset = CIFAR10(root='/home/disk2/hulai/Datasets/CIFAR10', train=True, download=True, transform = transform_train)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoaderX(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
                                              pin_memory=True, num_workers=4, sampler=train_sampler)

    test_dataset = CIFAR10(root='/home/disk2/hulai/Datasets/CIFAR10', train=False, download=True, transform = transform_test)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = DataLoaderX(test_dataset, batch_size=batch_size, shuffle=False,
                                              pin_memory=True, num_workers=4, sampler=test_sampler)



    best_acc = 0
    best_Epoch = 0
    # set total epoch
    total_epoch = 3
    end_epoch = start_epoch + total_epoch
    for epoch in range(start_epoch, end_epoch):
        model.train()
        total_loss = 0
        total = 0
        # correct = 0
        correct = torch.zeros(1).to(rank)
        train_sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(rank)
            target = target.to(rank)
            
            output = model(data)
            loss = criterion(output, target)
            
            # backward
            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            optimizer.step()

            reduced_loss = reduce_tensor(loss.data, world_size)



            # total_loss += loss.item()
            total_loss += reduced_loss.item()
            _, predicted = output.max(1)

            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            reduced_correct = reduce_tensor(correct, world_size)

            training_loss = (total_loss/(batch_idx+1))
            training_acc  = (reduced_correct/total).item()


            if batch_idx % 10 == 0:
            # if rank == 0 & batch_idx % 10 == 0:
                print('<===== Train =====> Epoch: [{}/{}]    training_loss = {:8.5f}    training_clean_acc = {:8.5f} \
                    training_batchsize = {}'.format(epoch, end_epoch-1, training_loss, training_acc, batch_size))

        dist.barrier()
        model.eval()
        total_test_loss = 0
        total_test = 0
        correct_test = 0
        test_sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(rank)
            target = target.to(rank)
            
            output = model(data)
            loss = criterion(output, target)
            

            
            total_test_loss += loss.item()
            _, predicted = output.max(1)

            total_test += target.size(0)
            correct_test += predicted.eq(target).sum().item()

            test_loss = (total_test_loss/(batch_idx+1))
            test_acc  = (correct_test/total_test)


            if rank == 0 & batch_idx % 10 == 0:
                print('<===== Test =====> Epoch: [{}/{}]    test_loss = {:8.5f}    test_clean_acc = {:8.5f} \
                    test_batchsize = {}'.format(epoch, end_epoch-1, test_loss, test_acc, batch_size))
        dist.barrier()
        # checkpoint
        acc = 100.*correct_test/total_test
        print('Epoch = {} : Acc = {}'.format(epoch, acc))
        if (acc > best_acc) & (rank % world_size == 0):
            print('Saving model ...')
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch+1,
                'optimizer': optimizer.state_dict()
            }
            if not os.path.isdir('checkpoint1'):
                os.mkdir('checkpoint1')
            torch.save(state, './checkpoint/resnet_cifar10_checkpoint.pth')
            best_acc = acc
            best_Epoch = epoch
        if (rank == 0) & (epoch == total_epoch-1) :
            print("The accuracy at Epoch {} is best accuracy: {}".format(best_Epoch, best_acc))
        # dist.barrier()
    cleanup()    

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)



if __name__ == "__main__":
    # Specify the GPU used
    os.environ['CUDA_VISIBLE_DEVICES'] ='5,6'
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    run_demo(run, world_size)
