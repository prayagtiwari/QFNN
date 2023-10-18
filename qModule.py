import argparse
import datetime
import logging
import os
import random
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import torch
from torch.utils.data import DataLoader, dataset
from torch.autograd import Variable

from ExpDataset import MustardDataset
from hyper_mt import HyperModel
from m2model_train import M2Model



class Q_Model(nn.Module):
  def __init__(self,bszi):
    super().__init__()
    self.encoder= tq.GeneralEncoder([
        {'input_idx': [0], 'func': 'rx', 'wires': [0]},
        {'input_idx': [1], 'func': 'rx', 'wires': [1]},
        {'input_idx': [2], 'func': 'rx', 'wires': [2]},
        {'input_idx': [3], 'func': 'rx', 'wires': [3]},
        {'input_idx': [4], 'func': 'rx', 'wires': [4]},
        {'input_idx': [5], 'func': 'rx', 'wires': [5]},
      ])

    self.sar_L=nn.Linear(3,3)
    self.sen_L=nn.Linear(3,3)

    
    self.measure = tq.MeasureAll(tq.PauliZ)

    self.re_sar_L=nn.Linear(3,2)
    self.re_sen_L=nn.Linear(3,3)

  def forward(self,sa,sen):

    device=tq.QuantumDevice(n_wires=6,bsz=sa.shape[0],device='cuda')
    # sa,sen=self.classM(txt,img,audio)
    sa=self.sar_L(sa)
    sen=self.sen_L(sen)

    qin=torch.concat([sen,sa],1)
    self.encoder(device,qin)   

    tq.cx(device,[0,1])
    tq.cx(device,[1,2])
    tq.cx(device,[2,0])

    tq.cx(device,[3,4])
    tq.cx(device,[4,5])
    tq.cx(device,[5,3])


    x=self.measure(device)
    sar=x[:,3:6]

    sar=self.re_sar_L(sar)
    sen=x[:,0:3]

    sen=self.re_sen_L(sen)
    return sar,sen


def trainEval():

    batchsize = 500
    epochs = 40

    all_epochs=280

    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    
    stateSavePATH = f'result/state/{all_epochs}_Epochs_{now}/state.pt'
    modelSavePATH = f'result/state/{all_epochs}_Epochs_{now}/model.pt'

    data = MustardDataset(datatye='train')
    valData = MustardDataset(datatye='dev')

    
    data_loader = DataLoader(
        data,
        batch_size=batchsize,
        shuffle=True,
        pin_memory=True
        )
    val_loader = DataLoader(
        valData,
        batch_size=50,
        shuffle=True,
        pin_memory=True,
        )
    class_model=torch.load('result/state/80_Epochs_2023-06-24-12-51-18/model.pt')
    class_model.eval()
    model = Q_Model(bszi=batchsize).cuda()


    
    lossFun = nn.CrossEntropyLoss().cuda()
    lossFun_val = nn.CrossEntropyLoss().cuda()
    

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.05, weight_decay=0.015)
    

    total = sum([param.nelement() for param in model.parameters()])
    print(f"Number of parameter: {total}")

    train_step = 0
    os.mkdir(f'result/state/{all_epochs}_Epochs_{now}')
    for _ in range(epochs):
        logger.info('epoch:'+str(_))
        

        for batch in data_loader:
            train_step += 1
           
            model.train()
            textInput = batch[0][0].cuda().to(torch.float32)
            imageInput = batch[0][1].to(torch.float32).cuda()
            audioInput = batch[0][2].to(torch.float32).cuda()
          
            sarLabel = batch[1][0].to(torch.float32).cuda()
            sentLabel = batch[1][1].to(torch.float32).cuda()

            with torch.no_grad():
               sar, sent= class_model.Qin(textInput, imageInput,audioInput)

            sar,sent=model(sar,sent)

            sarArgmax = torch.argmax(sarLabel, dim=-1)
            sentArgmax = torch.argmax(sentLabel, dim=-1)
            # emoArgmax = torch.argmax(emoLabel, dim=-1)
            # logger.info(f'sar:{sarArgmax} sent:{sentArgmax} emo:{emoArgmax}\n')
            loss1 = lossFun(sar, sarArgmax)
            loss2 = lossFun(sent, sentArgmax)
            # loss3 = lossFun(emo, emoArgmax)
            loss = (loss1 + loss2 )/2
            
            loss.requires_grad_(True)

            logger.info('loss1:%f loss2:%f loss:%f\n'
                        % (loss1.item(),
                            loss2.item(),
                            loss.item()))

            if train_step%(epochs-1) == 0:
                label_sar = np.argmax(
                    sarLabel.cpu().detach().numpy(), axis=-1)
                label_sent = np.argmax(
                    sentLabel.cpu().detach().numpy(), axis=-1)
                # label_emo = np.argmax(
                #     emoLabel.cpu().detach().numpy(), axis=-1)
                pred_sar = np.argmax(
                    sar.cpu().detach().numpy(), axis=1)
                pred_sent = np.argmax(
                    sent.cpu().detach().numpy(), axis=1)

                
                
                
                
                
                
                sar_f1 = f1_score(label_sar, pred_sar, average='micro')
                sent_f1 = f1_score(label_sent, pred_sent, average='micro')
                # emo_f1 = f1_score(label_emo, pred_emo, average='micro')
                sar_acc = accuracy_score(
                    label_sar, pred_sar)
                sent_acc = accuracy_score(
                    label_sent, pred_sent)

                logger.info(('train result sar-f1:%f sent-f1:%f ' +
                            'sar-acc:%f sent-acc:%f \n')
                            % (sar_f1, sent_f1,
                            sar_acc, sent_acc))

                model.eval()
                with torch.no_grad():
                    outputsar, outputsent, outputemo = [], [], []
                    tarsar, tarsent, taremo = [], [], []
                    for batch in val_loader:
                        textInput = batch[0][0].cuda().to(torch.float32)
                        imageInput = batch[0][1].cuda().to(torch.float32)
                        audioInput = batch[0][2].cuda().to(torch.float32)
                       
                        sarLabel = batch[1][0].to(torch.float32).cuda()
                        sentLabel = batch[1][1].to(torch.float32).cuda()
                
                        with torch.no_grad():
                           sar, sent=class_model.Qin(textInput, imageInput,audioInput)

                        
                        sar,sent=model(sar,sent)


                        sarArgmax = torch.argmax(sarLabel, dim=-1)
                        sentArgmax = torch.argmax(sentLabel, dim=-1)
              
                        loss1_val = lossFun_val(sar, sarArgmax)
                        loss2_val = lossFun_val(sent, sentArgmax)

                        loss_val = (loss1_val + loss2_val)/2

                        logger.info('val loss1:%f loss2:%f loss:%f\n'
                        % (loss1_val.item(),
                            loss2_val.item(),
                            loss_val.item()))

                        label_sar = np.argmax(
                            sarLabel.cpu().detach().numpy(), axis=-1)
                        label_sent = np.argmax(
                            sentLabel.cpu().detach().numpy(), axis=-1)
          
                        pred_sar = np.argmax(
                            sar.cpu().detach().numpy(), axis=1)
                        pred_sent = np.argmax(
                            sent.cpu().detach().numpy(), axis=1)
              
                        outputsar.append(pred_sar)
                        outputsent.append(pred_sent)
            
                        tarsar.append(label_sar)
                        tarsent.append(label_sent)
        

                    outputsar = np.concatenate(
                        np.array(outputsar),dtype=np.int64)
                    outputsent = np.concatenate(
                        np.array(outputsent),dtype=np.int64)
           
                    tarsar = np.concatenate(
                        np.array(tarsar),dtype=np.int64)
                    tarsent = np.concatenate(
                        np.array(tarsent),dtype=np.int64)
    

                    
                    sar_f1 = f1_score(
                        tarsar, outputsar, average='micro')
                    sent_f1 = f1_score(
                        tarsent, outputsent, average='micro')
     
                    sar_acc = accuracy_score(
                        tarsar, outputsar)
                    sent_acc = accuracy_score(
                        tarsent, outputsent)

                    logger.info(('val-result sar-f1:%f sent-f1:%f' +
                                ' sar-acc:%f sent-acc:%f\n')
                                % (sar_f1, sent_f1, 
                                sar_acc, sent_acc))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        torch.save(model.state_dict(), stateSavePATH)
        torch.save(model, modelSavePATH)

    
    
def testModel(modelPATH):
    testData = MustardDataset(datatye='test')
    batchsize = 32
    data_loader = DataLoader(
        testData,
        batch_size=batchsize,
        shuffle=True,
        pin_memory=True
        )
    classmodel=torch.load(ClassModelPath)
    
    model=torch.load(modelPATH)
    model.eval()
    classmodel.eval()
    with torch.no_grad():
        outputsar, outputsent, outputemo = [], [], []
        tarsar, tarsent, taremo = [], [], []
        for batch in data_loader:
            textInput = batch[0][0].cuda().to(torch.float32)
            imageInput = batch[0][1].cuda().to(torch.float32)
            wavInput = batch[0][2].cuda().to(torch.float32)

            
            sarLabel = batch[1][0].to(torch.float32).cuda()
            sentLabel = batch[1][1].to(torch.float32).cuda()
            emoLabel = batch[1][2].to(torch.float32).cuda()
            with torch.no_grad():
                sar, sent=classmodel(textInput, wavInput)
            inq=torch.cat((sar,sent),dim=1).cuda()
            sar,sent=model(inq)


            label_sar = np.argmax(
                sarLabel.cpu().detach().numpy(), axis=-1)
            label_sent = np.argmax(
                sentLabel.cpu().detach().numpy(), axis=-1)
            
            pred_sar = np.argmax(
                sar.cpu().detach().numpy(), axis=1)
            pred_sent = np.argmax(
                sent.cpu().detach().numpy(), axis=1)
        
            outputsar.append(pred_sar)
            outputsent.append(pred_sent)
            
            tarsar.append(label_sar)
            tarsent.append(label_sent)


        outputsar = np.concatenate(
            np.array(outputsar, dtype=object))
        outputsent = np.concatenate(
            np.array(outputsent, dtype=object))
        
        tarsar = np.concatenate(
            np.array(tarsar, dtype=object))
        tarsent = np.concatenate(
            np.array(tarsent, dtype=object))
        
        

        sar_f1 = f1_score(
            tarsar, outputsar, average='micro')
        sent_f1 = f1_score(
            tarsent, outputsent, average='micro')
       
        sar_acc = accuracy_score(
            tarsar, outputsar)
        sent_acc = accuracy_score(
            tarsent, outputsent)
    
        print('test tarsar:', tarsar)
        print('test outputsar:', outputsar)
        print('test tarsent:', tarsent)
        print('test outputsent:', outputsent)
        logger.info(('test-result sar-f1:%f sent-f1:%f' +
                    'sar-acc:%f sent-acc:%f \n')
                    % (sar_f1, sent_f1,
                    sar_acc, sent_acc))

  

if __name__ == "__main__":
    logging.basicConfig(filemode='w')
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(
        "result/log/new_log.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info("**********Start print log**********")
    

    trainEval()
    # testModel('result/state/10_Epochs_2023-06-18-05-41-09/model.pt')