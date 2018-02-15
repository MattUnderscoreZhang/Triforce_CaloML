from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda
import numpy as np
import h5py
import torch

import os
from torch import cat
from torch import from_numpy

from multiprocessing import Process, Pool

execfile("gamma_pi0_loader.py")

OutPath='test/'

def objective(params, GENERATOR_ID):
# define the model
    print params
    depth, width = params
    learning_rate=1.0e-3
    decay_rate=0.0e-6

    class ResBlock(nn.Module):
        def __init__(self, NumChannels): 
            super(ResBlock, self).__init__()
            self.conv0 = nn.Conv3d(NumChannels, NumChannels, 3, stride=1, padding=1)
            #self.bn0 = nn.BatchNorm3d(NumChannels)
            self.conv1 = nn.Conv3d(NumChannels, NumChannels, 3, stride=1, padding=1)
            #self.bn1 = nn.BatchNorm3d(NumChannels)
            self.selu0 = nn.SELU()
            self.selu1 = nn.SELU()

        def forward(self, x):
            #y = self.bn0(x)
            y = self.conv0(x)
            #y = self.bn1(y)
            y = self.selu0(y)
            y = self.conv1(y)
            return self.selu1(torch.add(y, x))


    class ResNet(nn.Module):
        def __init__(self):
            super(ResNet, self).__init__()
            self.conv0 = nn.Conv3d(1, 32, 3, stride=1, padding=1)
            #self.norm0 = nn.BatchNorm3d(64)
            #self.conv00 = nn.Conv3d(64, 64, 3, stride=1, padding=1)
            self.conv1 = nn.Conv3d(32, 64, 3, stride=2)
            #self.norm1 = nn.BatchNorm3d(96)
            #self.conv11 = nn.Conv3d(96, 96, 3, stride=1, padding=1)
            self.conv2 = nn.Conv3d(64, 96, 3, stride=2, padding=1)
            #self.norm2 = nn.BatchNorm3d(128)
            #self.conv22 = nn.Conv3d(128, 128, 3, stride=1, padding=1)
            self.conv3 = nn.Conv3d(96, 128, 3, stride=2, padding=1)
            #self.norm3 = nn.BatchNorm3d(192)
            #self.conv33 = nn.Conv3d(192, 192, 3, stride=1, padding=1)
            self.conv4 = nn.Conv3d(128, 192, 3, stride=1)
            #self.norm4 = nn.BatchNorm3d(width)
            #self.conv5 = nn.Conv3d(width, width, 3, stride=1)
            self.fc1 = nn.Linear(width, width)
            self.fc2 = nn.Linear(width, 2)
            #self.norm = nn.BatchNorm1d(width)


            self.selu0 = nn.SELU()
            self.selu1 = nn.SELU()
            self.selu2 = nn.SELU()
            self.selu3 = nn.SELU()
            self.selu4 = nn.SELU()
            self.selu5 = nn.SELU()
            self.selu6 = nn.SELU()
            self.block0 = self.build_layer(4, 32)
            self.block1 = self.build_layer(4, 64)
            self.block2 = self.build_layer(4, 96)

        def build_layer(self, NumLayers, NumChannels):
            layers = []
            for _ in range(NumLayers):
                layers.append(ResBlock(NumChannels))
            return nn.Sequential(*layers)


        def forward(self, x):
            x = x.view(-1, 1, 25, 25, 25)
        
            x = self.selu0(x)
            x = self.conv0(x)
            x = self.selu1(x)

            x = self.block0(x)

            x = self.conv1(x)
            x = self.selu2(x)

            x = self.block1(x)

            x = self.conv2(x)
            x = self.selu3(x)
            
            #x = self.rselu(self.norm2(x))

            x = self.block2(x)
         
            x = self.conv3(x)
            x = self.selu4(x)
            #x = self.rselu(self.norm3(x))

            x = self.conv4(x)
            x = self.selu5(x)

            x = x.view(-1, width)
            #x = self.norm(x)
            x = self.fc1(x)
            #x = self.norm(x)
            x = self.selu6(x)
            x = self.fc2(x)
            #x = self.softmax(x)
            return x


    from torch import load

    net = nn.DataParallel(ResNet(), device_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    #net = ResNet()

# load previous model
    
    #net.load_state_dict(load("test/savedmodel_depth_15-width_192"))

    net.cuda()

    import torch.optim as optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    net.train()
    criterion = nn.CrossEntropyLoss().cuda()
    #optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=decay_rate, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=decay_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, min_lr=1.0e-3, patience=10, factor=0.1, threshold = 1.0e-4)


    loss_history = []
    epoch_end_relative_error_history = []


    epoch_num=50

    #import pdb;pdb.set_trace()
    net.train()
# main process for training
    prev_val_loss = 0.0
    stag_break_count = 0
    early_stop_count = 0
    prev_epoch_end_val_loss = 0.0
    epoch_end_val_loss = 0.0

    #train_generator.start()
    train_loader = train_generator.generators[GENERATOR_ID].generate()
    
    #val_generator.start()
    val_loader = val_generator.generators[GENERATOR_ID].generate()
    
    running_loss = 0.0
    val_loss = 0.0
    i = 0
    import itertools
    for train_data, val_data in itertools.izip(train_loader, val_loader):
        #inputs, labels = data
        #ECAL, HCAL, labels = Variable(inputs[0].cuda()), Variable(inputs[1].cuda()), Variable(labels.cuda())
        net.train()
        ECAL, _, labels = train_data
        ECAL = np.swapaxes(ECAL,1,3)
        ECAL, labels = Variable(from_numpy(ECAL).cuda()), Variable(from_numpy(labels).long().cuda())
        optimizer.zero_grad()
        ECAL = 50000.0*ECAL
        outputs = net(ECAL)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]



        net.eval()
        ECAL, _, labels = val_data
        ECAL = np.swapaxes(ECAL,1,3)
        ECAL, labels = Variable(from_numpy(ECAL).cuda()), Variable(from_numpy(labels).long().cuda())
        ECAL = 50000.0*ECAL
        val_outputs = net(ECAL)
        validation_loss = criterion(val_outputs, labels)
        val_loss += validation_loss.data[0]

        if i % 20 == 19:
            running_loss /= 20
            val_loss /= 20
            print('[%d, %5d, %5d] loss: %.10f' %
                    (GENERATOR_ID, i/4000 + 1, i%4000 + 1, running_loss)),
            print('    val loss: %.10f' %
                    (val_loss)),
            relative_error = (val_loss-prev_val_loss)/float(val_loss)
            print('    relative error: %.10f' %
                    (relative_error)),

            if(val_loss < 0.10):
                break
            scheduler.step(val_loss)
            if(relative_error>0.01 and i!=0):
                early_stop_count+=1
                if(early_stop_count>5):
                    break
            else:
                early_stop_count=0
            
            print('    early stop count: %d' %
                    (early_stop_count))
            
            loss_history.append([GENERATOR_ID, i/4000 + 1, i%4000 + 1, running_loss, val_loss, relative_error, early_stop_count])
            
          #  if(i % 400==399):
          #      epoch_end_val_loss = val_loss
          #      epoch_end_relative_error = (epoch_end_val_loss-prev_epoch_end_val_loss)/float(epoch_end_val_loss)
          #      print('[%d] epoch_end_relative_error: %.10f' %
          #              (GENERATOR_ID, epoch_end_relative_error)),
          #      epoch_end_relative_error_history.append([GENERATOR_ID, i/4000 + 1, i%4000 + 1, epoch_end_relative_error])

          #      if(epoch_end_relative_error > -0.005 and i/4000!=0):
          #          stag_break_count+=1
          #          if(stag_break_count>0):
          #              break
          #      else:
          #          stag_break_count=0
          #      print('    stag_break_count: %d' %
          #              (stag_break_count))
          #      prev_epoch_end_val_loss = epoch_end_val_loss

          #  
            prev_val_loss = val_loss
            running_loss = 0.0
            val_loss = 0.0
        i+=1

    #train_generator.hard_stop()
    #val_generator.hard_stop()
        #break;

    loss_history = np.array(loss_history)
    epoch_end_relative_error_history = np.array(epoch_end_relative_error_history)
    with h5py.File(OutPath+"loss_history-depth_"+str(depth)+"-width_"+str(width)+".h5", 'w') as loss_file, h5py.File(OutPath+"epoch_end_relative_error_history-depth_"+str(depth)+"-width_"+str(width)+".h5", 'w') as epoch_end_relative_error_history_file:
        loss_file.create_dataset("loss", data=loss_history)
        epoch_end_relative_error_history_file.create_dataset("relative_error", data=epoch_end_relative_error_history)

    from torch import save
    save(net.state_dict(), OutPath+"savedmodel_depth_"+str(depth)+"-width_"+str(width))

    print('Finished Training')

# Analysis
    from torch import max

    correct = 0
    total = 0
    #test_generator.start()
    test_loader = test_generator.generators[GENERATOR_ID].generate()
    test_count = 0
    for test_index, test_data in enumerate(test_loader):
        test_count += 1
        #images, labels = data
        #ECAL, HCAL, labels = Variable(images[0].cuda()), Variable(images[1].cuda()), labels.cuda()
        #outputs = net(ECAL, HCAL)
        ECAL, _, labels = test_data
        ECAL = np.swapaxes(ECAL,1,3)
        ECAL, labels = Variable(from_numpy(ECAL).cuda()), from_numpy(labels).long().cuda()
        ECAL = 50000.0*ECAL
        outputs = net(ECAL)
        _, predicted = max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

        if(test_count >= 300):
            break;
    #test_generator.hard_stop()
    print('Accuracy of the network on test images: %f %%' % (
            100 * float(correct) / total))

    return (float(correct) / total)*100.0


def run(pid):
    width_min=256
    width_max=511
    depth = 30 
    depth_min = 5 
    depth_max = 5 

    accuracies = []

    with torch.cuda.device(pid%num_gpus):
        for width in range(width_min + pid, width_max + 1, num_gen):
        #import rpdb; rpdb.set_trace()
            accuracy = objective((depth, width), pid)

            accuracies.append([width, accuracy])

    with h5py.File(OutPath+"accuracies_depth_range_"+str(depth_min)+"-"+str(depth_max)+"_width_range_"+str(width_min)+"-"+str(width_max)+"_"+str(pid)+".h5", 'w') as write_accuracies:
        write_accuracies.create_dataset('accuracies', data=np.array(accuracies))


if __name__ == '__main__':

    num_of_processes = num_gen


    #objective.index_of_call=0


    train_generator.start()
    val_generator.start()
    test_generator.start()

    #worker_pool = Pool(processes=num_of_processes)
    accuracy = objective((18, 192), 0)
    #worker_pool.map(run, range(num_of_processes))


    train_generator.hard_stop()
    val_generator.hard_stop()
    test_generator.hard_stop()
