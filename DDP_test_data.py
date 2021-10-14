import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from Resnet_refine import ResNet_refine

from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

    test_dataset = CIFAR10(root='/home/disk2/hulai/Datasets/CIFAR10', train=False, download=True, transform = transform_test)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)


    model = ResNet_refine('resnet18', False, 10).to(rank)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    start_epoch = 0
    resume = 0
    if resume == 0:
        print('resuming from checkpoint...')
        checkpoint = torch.load('./checkpoint/resnet_cifar10_checkpoint.pth')
        model.load_state_dict(checkpoint['model'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])

    best_acc = 0
    best_Epoch = 0
    total_epoch = 3
    end_epoch = start_epoch + total_epoch
    for epoch in range(start_epoch, end_epoch):
        model.train()
        total_loss = 0
        total = 0
        correct = 0
        train_sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(rank)
            target = target.to(rank)
            
            output = model(data)
            loss = criterion(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = output.max(1)

            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            training_loss = (total_loss/(batch_idx+1))
            training_acc  = (correct/total)


            if batch_idx % 10 == 0:
                print('<===== Train =====> Epoch: [{}/{}]    training_loss = {:8.5f}    training_clean_acc = {:8.5f} \
                    training_batchsize = {}'.format(epoch, end_epoch-1, training_loss, training_acc, batch_size))


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
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_test_loss += loss.item()
            _, predicted = output.max(1)

            total_test += target.size(0)
            correct_test += predicted.eq(target).sum().item()

            test_loss = (total_loss/(batch_idx+1))
            test_acc  = (correct_test/total_test)


            if batch_idx % 10 == 0:
                print('<===== Test =====> Epoch: [{}/{}]    test_loss = {:8.5f}    test_clean_acc = {:8.5f} \
                    test_batchsize = {}'.format(epoch, end_epoch-1, test_loss, test_acc, batch_size))
        
        # checkpoint
        acc = 100.*correct_test/total_test
        print('Epoch = {} : Acc = {}'.format(epoch, acc))
        if (acc > best_acc) & (rank == 0):
            print('Saving model ...')
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch+1,
                'optimizer': optimizer.state_dict()
            }
            # if not os.path.isdir('checkpoint'):
            #     os.mkdir('checkpoint')
            torch.save(state, './checkpoint/resnet_cifar10_checkpoint.pth')
            best_acc = acc
            best_Epoch = epoch
        if (rank == 0) & (epoch == total_epoch-1) :
            print("The accuracy at Epoch {} is best accuracy: {}".format(best_Epoch, best_acc))
    cleanup()    

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)



if __name__ == "__main__":
    # 指定使用的显卡
    os.environ['CUDA_VISIBLE_DEVICES'] ='0,2,3'
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    run_demo(run, world_size)

    # mp.spawn(run,
    #          args=(world_size,),
    #          nprocs=world_size,
    #          join=True)