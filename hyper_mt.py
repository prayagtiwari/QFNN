import datetime
from os import TMP_MAX
import os
from ExpDataset import MustardDataset
import torch
from torch import nn
from torch.nn import Module
from torch.nn import Dropout, Flatten, Linear,\
    Softmax, GRU, AvgPool3d, MultiheadAttention
from torch.utils.data import DataLoader, dataset
from transformers import AlbertModel, AlbertConfig
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from torch.autograd import Variable
import logging
import qModule

#txt:torch.Size([4, 24, 768])
#img:torch.Size([4, 24, 56,56])
#au:torch.Size([4,24,128])

import torch
import torch.nn as nn

# class ComplexMultiplier(nn.Module):
#     def __init__(self, input_size):
#         super(ComplexMultiplier, self).__init__()
#         self.r = nn.Parameter(torch.Tensor(input_size))  # 可训练参数 r
#         self.t = nn.Parameter(torch.Tensor(input_size))  # 可训练参数 t

#         # 初始化参数 r 和 t
#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.uniform_(self.r)  # 使用均匀分布初始化 r
#         nn.init.uniform_(self.t)  # 使用均匀分布初始化 t

#     def forward(self, input):
#         real = torch.cos(self.t) * self.r
#         imag = torch.sin(self.t) * self.r
#         complex_weights = torch.complex(real, imag)  # 构造复数权重

#         return torch.mul(input, complex_weights)


class HyperModel(Module):
    def __init__(self, batchsize):
        super(HyperModel, self).__init__()


        self.img_maxpool_layer = nn.MaxPool2d(kernel_size=4, stride=4)
        self.img_batchnorm_layer = nn.BatchNorm2d(24)
        self.batchsize=batchsize

        self.textBiGRU = GRU(
            input_size=768,
            hidden_size=64,
            num_layers=1,
            batch_first=True, 
            bidirectional=True)

        self.imageBiGRU = GRU(
            input_size=196,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True)

        self.audioBiGRU = GRU(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True)

        
        

        
        self.textEncodeAtt = MultiheadAttention(embed_dim=128, num_heads=1)
        self.imageEncodeAtt = MultiheadAttention(embed_dim=128, num_heads=1)
        self.audioEncodeAtt = MultiheadAttention(embed_dim=128, num_heads=1)


        self.targetWeight = nn.Parameter(torch.Tensor([0.5]))
        self.contextWeight = nn.Parameter(torch.Tensor([0.5]))

        self.interTtoI = MultiheadAttention(
            128, 4)
        self.interItoT = MultiheadAttention(
            128, 4)
        self.interTtoA = MultiheadAttention(
            128, 4)
        self.interAtoT = MultiheadAttention(
            128, 4)
        self.interAtoI = MultiheadAttention(
            128, 4)
        self.interItoA = MultiheadAttention(
            128, 4)
        

        
        self.sentimentGRU = GRU(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=False)
        # self.emotionGRU = GRU(
        #     input_size=128,
        #     hidden_size=128,
        #     num_layers=1,
        #     batch_first=False)
        self.sarcasmGRU = GRU(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=False)
        self.sentimentAtt = MultiheadAttention(128, 1)
        # self.emotionAtt = MultiheadAttention(128, 1)
        self.sarcasmAtt = MultiheadAttention(128, 1)

        self.directSent = nn.Linear(
            in_features=18432, out_features=3)
        self.sentBN = nn.BatchNorm1d(18432)

        # self.directEmo = nn.Linear(
        #     in_features=4864, out_features=9)
        # self.emoBN = nn.BatchNorm1d(4864)
        
        self.directSar = nn.Linear(
            in_features=37376, out_features=3)
        self.sarBN = nn.BatchNorm1d(37376)

        self.re_sar=nn.Linear(3,2)

        # self.sarScaleAct = nn.Tanh()
        # self.dropout = nn.Dropout(p=0.4)

        
    def Qin(self, text, image, audio):
        
        contextsize = text.shape[1]
        text = text
        image = image
        audio=audio

        textChunks = torch.chunk(text, contextsize, 1)
        imageChunks = torch.chunk(image, contextsize, 1)
        audioChunks = torch.chunk(audio, contextsize, 1)
        tChunkList = []
        iChunkList = []
        aChunkList = []
        thnList = []
        ihnList = []
        ahnList = []
        tEncoderAttWeight = []
        iEncoderAttWeight = []
        aEncoderAttWeight = []

        
        for tChunk in textChunks:
            tChunk = torch.squeeze(tChunk, dim=1)

            tChunk, thn = self.textBiGRU(tChunk)
            # tChunk, thn = self.textBiGRU2(tChunk)

            # tChunk=tChunk.reshape(tChunk.shape[0],3,-1)

            query = torch.transpose(tChunk, dim0=0, dim1=1)
            key = torch.transpose(tChunk, dim0=0, dim1=1)
            value = torch.transpose(tChunk, dim0=0, dim1=1)

            #(24,b,32)
            textAttnOutput, textAttnOutputWeights = \
                self.textEncodeAtt(query, key, value)
            
            tChunkList.append(textAttnOutput)
            

        for iChunk in imageChunks:
           
            iChunk = torch.squeeze(iChunk, dim=1)

            iChunk=self.img_maxpool_layer(iChunk)
            iChunk=self.img_batchnorm_layer(iChunk) 
        
            iChunk=iChunk.view(iChunk.shape[0],iChunk.shape[1],-1)
            #ichunk (b,24,14*14)
            iChunk, ihn = self.imageBiGRU(iChunk)
            # iChunk, ihn = self.imageBiGRU2(iChunk)

            # iChunk=iChunk.reshape(iChunk.shape[0],3,-1)
            ihnList.append(ihn)

            query = torch.transpose(iChunk, dim0=0, dim1=1)
            key = torch.transpose(iChunk, dim0=0, dim1=1)
            value = torch.transpose(iChunk, dim0=0, dim1=1)
            #imageAttnOutput (24,b,32)
            imageAttnOutput, imageAttnOutputWeights = \
                self.imageEncodeAtt(query, key, value)
            #imageAttnOutput (24,b,32)

            iChunkList.append(imageAttnOutput)
            iEncoderAttWeight.append(imageAttnOutputWeights)
            
            
            

        for aChunk in audioChunks:
            
            aChunk = torch.squeeze(aChunk, dim=1)
            
            aChunk, ahn = self.audioBiGRU(aChunk)
            # aChunk, ahn = self.audioBiGRU2(aChunk)

            # aChunk=aChunk.reshape(aChunk.shape[0],3,-1)

            ahnList.append(ahn)
            query = torch.transpose(aChunk, dim0=0, dim1=1)
            key = torch.transpose(aChunk, dim0=0, dim1=1)
            value = torch.transpose(aChunk, dim0=0, dim1=1)
            
            
            audioAttnOutput, audioAttnOutputWeights = \
                self.audioEncodeAtt(query, key, value)
        
            aChunkList.append(audioAttnOutput)
            aEncoderAttWeight.append(audioAttnOutputWeights)

        targetText = tChunkList[-1]
        targetImage = iChunkList[-1]
        targetAudio = aChunkList[-1]

        '''
        text
        '''
        tConAtt = [torch.exp(
            torch.tanh(targetText+tContext)+1e-7)
            for tContext in tChunkList[:-1]
        ]
        tConAttSum = tConAtt[0] + tConAtt[1] + tConAtt[2]
        tIntraAttOutput = [
            (tConAtt[i]/tConAttSum)*tChunkList[i] for i in range(3)
        ]
        tIntraAttOutput = tIntraAttOutput[0] +\
            tIntraAttOutput[1]+tIntraAttOutput[2]
        
        
        tIntraAttOutput = targetText + \
            torch.tanh(
                self.targetWeight * targetText
                + self.contextWeight * tIntraAttOutput
            )
       
        '''
        text
        '''

        '''
        image
        '''
        iConAtt = [torch.exp(
            torch.tanh(targetImage+iContext)+1e-7)
            for iContext in iChunkList[:-1]
        ]
        iConAttSum = iConAtt[0] + iConAtt[1] + iConAtt[2]
        
        iIntraAttOutput = [
            (iConAtt[i]/iConAttSum)*iChunkList[i] for i in range(3)
        ]
        iIntraAttOutput = iIntraAttOutput[0] +\
            iIntraAttOutput[1]+iIntraAttOutput[2]
        iIntraAttOutput = targetImage + \
            torch.tanh(
                self.targetWeight * targetImage
                + self.contextWeight * iIntraAttOutput
            )
        
        '''
        image
        '''

        '''
        audio
        '''
        aConAtt = [torch.exp(
            torch.tanh(targetAudio+aContext)+1e-7)
            for aContext in aChunkList[:-1]
        ]

        aConAttSum = aConAtt[0]+aConAtt[1]+aConAtt[2]

        aIntraAttOutput = [
            (aConAtt[i]/aConAttSum)*aChunkList[i] for i in range(3)
        ]

        aIntraAttOutput = aIntraAttOutput[0] +\
            aIntraAttOutput[1]+aIntraAttOutput[2]
        
        aIntraAttOutput = targetAudio +\
            torch.tanh(
                self.targetWeight * targetAudio
                + self.contextWeight * aIntraAttOutput
            )
    
        '''
        audio
        '''
        

        
        textKey, textValue, textQuery = \
            tIntraAttOutput, tIntraAttOutput, tIntraAttOutput
        imageKey, imageValue, imageQuery = \
            iIntraAttOutput, iIntraAttOutput, iIntraAttOutput
        audioKey, audioValue, audioQuery = \
            aIntraAttOutput, aIntraAttOutput, aIntraAttOutput
        
        interTtoI, _ = self.interTtoI(
            query=textQuery, key=imageKey, value=imageValue)
        interItoT, _ = self.interItoT(
            query=imageQuery, key=textKey, value=textValue)
        interTtoA, _ = self.interTtoA(
            query=textQuery, key=audioKey, value=audioValue)
        interAtoT, _ = self.interAtoT(
            query=audioQuery, key=textKey, value=textValue)
        interItoA, _ = self.interItoA(
            query=imageQuery, key=audioKey, value=audioValue)
        interAtoI, _ = self.interAtoI(
            query=audioQuery, key=imageKey, value=imageValue)
        
        ITCat = torch.cat([interTtoI, interItoT], dim=0)
        ATCat = torch.cat([interTtoA, interAtoT], dim=0)
        IACat = torch.cat([interAtoI, interItoA], dim=0)
        
        
        
        
        catFeature = torch.cat([ITCat, ATCat, IACat], dim=0)
        
        
        
        

        
        flatten = nn.Flatten()
        
        sentGRU, sentHidden = self.sentimentGRU(catFeature)
        sentAtt, _ = self.sentimentAtt(sentGRU, sentGRU, sentGRU)
        sentAtt = torch.transpose(sentAtt, dim0=0, dim1=1)
        sentAttFlatten = flatten(sentAtt)
        
        sentOutput = self.sentBN(sentAttFlatten)
        sentOutput = self.directSent(sentOutput)
        

        #topk的作用：取出每一行最大的值，返回值为两个，第一个为最大值，第二个为最大值的索引
        #topk的参数：第一个为要取出的最大值的个数，第二个为取出的维度
        sentTop1 = torch.topk(sentOutput, 1)[1]
        #repeat：将sentTop1中的每个元素重复128次
        sentTop1 = sentTop1.repeat(1, 128)
        sentTop1 = torch.unsqueeze(sentTop1, 1)
        sentTop1 = torch.transpose(sentTop1, dim0=0, dim1=1)

        sarInput = torch.cat([catFeature, sentHidden, sentTop1], dim=0)
        sarGRU, sarHidden = self.sarcasmGRU(sarInput)
        sarAtt, _ = self.sarcasmAtt(sarGRU, sarGRU, sarGRU)
        sarAtt = torch.transpose(sarAtt, dim0=0, dim1=1)
        sarAttFlatten = torch.cat([flatten(sarAtt), flatten(sarAtt)], dim=1)
        
        sarOutput = self.sarBN(sarAttFlatten)
        sarOutput = self.directSar(sarOutput)
        

        
        return sarOutput, sentOutput
    
    def forward(self, text, image, audio):

       
        contextsize = text.shape[1]
        text = text
        image = image
        audio=audio

        textChunks = torch.chunk(text, contextsize, 1)
        imageChunks = torch.chunk(image, contextsize, 1)
        audioChunks = torch.chunk(audio, contextsize, 1)
        tChunkList = []
        iChunkList = []
        aChunkList = []
        thnList = []
        ihnList = []
        ahnList = []
        tEncoderAttWeight = []
        iEncoderAttWeight = []
        aEncoderAttWeight = []

        
        for tChunk in textChunks:
            tChunk = torch.squeeze(tChunk, dim=1)

            tChunk, thn = self.textBiGRU(tChunk)
            # tChunk, thn = self.textBiGRU2(tChunk)

            # tChunk=tChunk.reshape(tChunk.shape[0],3,-1)

            query = torch.transpose(tChunk, dim0=0, dim1=1)
            key = torch.transpose(tChunk, dim0=0, dim1=1)
            value = torch.transpose(tChunk, dim0=0, dim1=1)

            #(24,b,32)
            textAttnOutput, textAttnOutputWeights = \
                self.textEncodeAtt(query, key, value)
            
            tChunkList.append(textAttnOutput)
            

        for iChunk in imageChunks:
           
            iChunk = torch.squeeze(iChunk, dim=1)

            iChunk=self.img_maxpool_layer(iChunk)
            iChunk=self.img_batchnorm_layer(iChunk) 
        
            iChunk=iChunk.view(iChunk.shape[0],iChunk.shape[1],-1)
            #ichunk (b,24,14*14)
            iChunk, ihn = self.imageBiGRU(iChunk)
            # iChunk, ihn = self.imageBiGRU2(iChunk)

            # iChunk=iChunk.reshape(iChunk.shape[0],3,-1)
            ihnList.append(ihn)

            query = torch.transpose(iChunk, dim0=0, dim1=1)
            key = torch.transpose(iChunk, dim0=0, dim1=1)
            value = torch.transpose(iChunk, dim0=0, dim1=1)
            #imageAttnOutput (24,b,32)
            imageAttnOutput, imageAttnOutputWeights = \
                self.imageEncodeAtt(query, key, value)
            #imageAttnOutput (24,b,32)

            iChunkList.append(imageAttnOutput)
            iEncoderAttWeight.append(imageAttnOutputWeights)
            
            
            

        for aChunk in audioChunks:
            
            aChunk = torch.squeeze(aChunk, dim=1)
            
            aChunk, ahn = self.audioBiGRU(aChunk)
            # aChunk, ahn = self.audioBiGRU2(aChunk)

            # aChunk=aChunk.reshape(aChunk.shape[0],3,-1)

            ahnList.append(ahn)
            query = torch.transpose(aChunk, dim0=0, dim1=1)
            key = torch.transpose(aChunk, dim0=0, dim1=1)
            value = torch.transpose(aChunk, dim0=0, dim1=1)
            
            
            audioAttnOutput, audioAttnOutputWeights = \
                self.audioEncodeAtt(query, key, value)
        
            aChunkList.append(audioAttnOutput)
            aEncoderAttWeight.append(audioAttnOutputWeights)

        targetText = tChunkList[-1]
        targetImage = iChunkList[-1]
        targetAudio = aChunkList[-1]

        '''
        text
        '''
        tConAtt = [torch.exp(
            torch.tanh(targetText+tContext)+1e-7)
            for tContext in tChunkList[:-1]
        ]
        tConAttSum = tConAtt[0] + tConAtt[1] + tConAtt[2]
        tIntraAttOutput = [
            (tConAtt[i]/tConAttSum)*tChunkList[i] for i in range(3)
        ]
        tIntraAttOutput = tIntraAttOutput[0] +\
            tIntraAttOutput[1]+tIntraAttOutput[2]
        
        
        tIntraAttOutput = targetText + \
            torch.tanh(
                self.targetWeight * targetText
                + self.contextWeight * tIntraAttOutput
            )
       
        '''
        text
        '''

        '''
        image
        '''
        iConAtt = [torch.exp(
            torch.tanh(targetImage+iContext)+1e-7)
            for iContext in iChunkList[:-1]
        ]
        iConAttSum = iConAtt[0] + iConAtt[1] + iConAtt[2]
        
        iIntraAttOutput = [
            (iConAtt[i]/iConAttSum)*iChunkList[i] for i in range(3)
        ]
        iIntraAttOutput = iIntraAttOutput[0] +\
            iIntraAttOutput[1]+iIntraAttOutput[2]
        iIntraAttOutput = targetImage + \
            torch.tanh(
                self.targetWeight * targetImage
                + self.contextWeight * iIntraAttOutput
            )
        
        '''
        image
        '''

        '''
        audio
        '''
        aConAtt = [torch.exp(
            torch.tanh(targetAudio+aContext)+1e-7)
            for aContext in aChunkList[:-1]
        ]

        aConAttSum = aConAtt[0]+aConAtt[1]+aConAtt[2]

        aIntraAttOutput = [
            (aConAtt[i]/aConAttSum)*aChunkList[i] for i in range(3)
        ]

        aIntraAttOutput = aIntraAttOutput[0] +\
            aIntraAttOutput[1]+aIntraAttOutput[2]
        
        aIntraAttOutput = targetAudio +\
            torch.tanh(
                self.targetWeight * targetAudio
                + self.contextWeight * aIntraAttOutput
            )
    
        '''
        audio
        '''
        

        
        textKey, textValue, textQuery = \
            tIntraAttOutput, tIntraAttOutput, tIntraAttOutput
        imageKey, imageValue, imageQuery = \
            iIntraAttOutput, iIntraAttOutput, iIntraAttOutput
        audioKey, audioValue, audioQuery = \
            aIntraAttOutput, aIntraAttOutput, aIntraAttOutput
        
        interTtoI, _ = self.interTtoI(
            query=textQuery, key=imageKey, value=imageValue)
        interItoT, _ = self.interItoT(
            query=imageQuery, key=textKey, value=textValue)
        interTtoA, _ = self.interTtoA(
            query=textQuery, key=audioKey, value=audioValue)
        interAtoT, _ = self.interAtoT(
            query=audioQuery, key=textKey, value=textValue)
        interItoA, _ = self.interItoA(
            query=imageQuery, key=audioKey, value=audioValue)
        interAtoI, _ = self.interAtoI(
            query=audioQuery, key=imageKey, value=imageValue)
        
        ITCat = torch.cat([interTtoI, interItoT], dim=0)
        ATCat = torch.cat([interTtoA, interAtoT], dim=0)
        IACat = torch.cat([interAtoI, interItoA], dim=0)
        
        
        
        
        catFeature = torch.cat([ITCat, ATCat, IACat], dim=0)
        
        
        
        

        
        flatten = nn.Flatten()
        
        sentGRU, sentHidden = self.sentimentGRU(catFeature)
        sentAtt, _ = self.sentimentAtt(sentGRU, sentGRU, sentGRU)
        sentAtt = torch.transpose(sentAtt, dim0=0, dim1=1)
        sentAttFlatten = flatten(sentAtt)
        
        sentOutput = self.sentBN(sentAttFlatten)
        sentOutput = self.directSent(sentOutput)
        

        #topk的作用：取出每一行最大的值，返回值为两个，第一个为最大值，第二个为最大值的索引
        #topk的参数：第一个为要取出的最大值的个数，第二个为取出的维度
        sentTop1 = torch.topk(sentOutput, 1)[1]
        #repeat：将sentTop1中的每个元素重复128次
        sentTop1 = sentTop1.repeat(1, 128)
        sentTop1 = torch.unsqueeze(sentTop1, 1)
        sentTop1 = torch.transpose(sentTop1, dim0=0, dim1=1)

        sarInput = torch.cat([catFeature, sentHidden, sentTop1], dim=0)
        sarGRU, sarHidden = self.sarcasmGRU(sarInput)
        sarAtt, _ = self.sarcasmAtt(sarGRU, sarGRU, sarGRU)
        sarAtt = torch.transpose(sarAtt, dim0=0, dim1=1)
        sarAttFlatten = torch.cat([flatten(sarAtt), flatten(sarAtt)], dim=1)
        
        sarOutput = self.sarBN(sarAttFlatten)
        sarOutput = self.directSar(sarOutput)
        sarOutput=self.re_sar(sarOutput)
        

        
        return sarOutput, sentOutput


def testModel(modelPATH):
    testData = MustardDataset(datatye='test')
    batchsize = 32
    data_loader = DataLoader(
        testData,
        batch_size=batchsize,
        shuffle=True,
        pin_memory=True
        )

    model = HyperModel(batchsize=batchsize).cuda()
    model=torch.load(modelPATH)
    model.eval()
    with torch.no_grad():
        # outputsar, outputsent, outputemo = [], [], []
        outputsar, outputsent = [], []
        tarsar, tarsent = [], []
        for batch in data_loader:
            textInput = batch[0][0].cuda().to(torch.float32)
            imageInput = batch[0][1].cuda().to(torch.float32)
            wavInput = batch[0][2].cuda().to(torch.float32)
            
            sarLabel = batch[1][0].to(torch.float32).cuda()
            sentLabel = batch[1][1].to(torch.float32).cuda()
            emoLabel = batch[1][2].to(torch.float32).cuda()
            # sar, sent, emo = \
            #     model(textInput, imageInput, wavInput)
            sar, sent= \
                model(textInput, imageInput, wavInput)

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
            # pred_emo = np.argmax(
            #     emo.cpu().detach().numpy(), axis=1)
            outputsar.append(pred_sar)
            outputsent.append(pred_sent)
            # outputemo.append(pred_emo)
            tarsar.append(label_sar)
            tarsent.append(label_sent)
            # taremo.append(label_emo)

        outputsar = np.concatenate(
            np.array(outputsar, dtype=object))
        outputsent = np.concatenate(
            np.array(outputsent, dtype=object))
        # outputemo = np.concatenate(
        #     np.array(outputemo, dtype=object))
        tarsar = np.concatenate(
            np.array(tarsar, dtype=object))
        tarsent = np.concatenate(
            np.array(tarsent, dtype=object))
        # taremo = np.concatenate(
        #     np.array(taremo, dtype=object))
        

        sar_f1 = f1_score(
            tarsar, outputsar, average='micro')
        sent_f1 = f1_score(
            tarsent, outputsent, average='micro')
        # emo_f1 = f1_score(
        #     taremo, outputemo, average='micro')
        sar_acc = accuracy_score(
            tarsar, outputsar)
        sent_acc = accuracy_score(
            tarsent, outputsent)
        # emo_acc = accuracy_score(
        #     taremo, outputemo)
        print('test tarsar:', tarsar)
        print('test outputsar:', outputsar)
        print('test tarsent:', tarsent)
        print('test outputsent:', outputsent)
        # print('test taremo:', taremo)
        # print('test outputemo:', outputemo)
        # logger.info(('test-result sar-f1:%f sent-f1:%f emo-f1:%f' +
        #             'sar-acc:%f sent-acc:%f emo-acc:%f\n')
        #             % (sar_f1, sent_f1, emo_f1,
        #             sar_acc, sent_acc, emo_acc))
        logger.info(('test-result sar-f1:%f sent-f1:%f' +
                    'sar-acc:%f sent-acc:%f\n')
                    % (sar_f1, sent_f1,
                    sar_acc, sent_acc))

modelPATH='result/state/40_Epochs_2023-06-24-00-39-28/model.pt'
def trainEval():

    batchsize = 500
    epochs = 80

    all_epochs=80

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
        batch_size=90,
        shuffle=True,
        pin_memory=True,
        )
    model = HyperModel(batchsize=batchsize).cuda()

    for m in model.modules():
        if isinstance(m, (Linear)):
            nn.init.xavier_uniform_(m.weight)
    
    # model=torch.load(modelPATH)
    
    
    

    
    lossFun = nn.CrossEntropyLoss().cuda()
    lossFun_val = nn.CrossEntropyLoss().cuda()
    

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=0.015)
    

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))

   
    os.mkdir(f'result/state/{all_epochs}_Epochs_{now}')
    train_step = 0
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
            # emoLabel = batch[1][2].to(torch.float32).cuda()

            textInput = Variable(textInput, requires_grad=True) 
            imageInput = Variable(imageInput, requires_grad=True)
            audioInput = Variable(audioInput, requires_grad=True)

            sarLabel = Variable(sarLabel, requires_grad=True)
            sentLabel = Variable(sentLabel, requires_grad=True)
            # emoLabel = Variable(emoLabel, requires_grad=True)

            sar, sent= model(textInput, imageInput, audioInput)
            sar = sar.to(torch.float32)
            sent = sent.to(torch.float32)
            # emo = emo.to(torch.float32)

            sarArgmax = torch.argmax(sarLabel, dim=-1)
            sentArgmax = torch.argmax(sentLabel, dim=-1)
            # emoArgmax = torch.argmax(emoLabel, dim=-1)
            loss1 = lossFun(sar, sarArgmax)
            loss2 = lossFun(sent, sentArgmax)
            # loss3 = lossFun(emo, emoArgmax)
            # loss = (loss1 + loss2 + loss3)/3
            loss = (loss1 + loss2)/3

            
            
            
            
            
            
            loss.requires_grad_(True)
            logger.info('loss1:%f loss2:%f loss:%f\n'
                        % (loss1.item(),
                            loss2.item(),
                            # loss3.item(),
                            loss.item()))

            if train_step%10 == 0:
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
                # pred_emo = np.argmax(
                #     emo.cpu().detach().numpy(), axis=1)

                
                
                
                
                
                
                sar_f1 = f1_score(label_sar, pred_sar, average='micro')
                sent_f1 = f1_score(label_sent, pred_sent, average='micro')
                # emo_f1 = f1_score(label_emo, pred_emo, average='micro')
                sar_acc = accuracy_score(
                    label_sar, pred_sar)
                sent_acc = accuracy_score(
                    label_sent, pred_sent)
                # emo_acc = accuracy_score(
                #     label_emo, pred_emo)
                # logger.info(('train result sar-f1:%f sent-f1:%f ' +
                #             'emo-f1:%f sar-acc:%f sent-acc:%f emo-acc:%f\n')
                #             % (sar_f1, sent_f1, emo_f1,
                #             sar_acc, sent_acc, emo_acc))
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
                        emoLabel = batch[1][2].to(torch.float32).cuda()
                        # sar, sent, emo = \
                        #     model(textInput, imageInput, audioInput)
                        sar, sent = model(textInput, imageInput, audioInput)

                        sarArgmax = torch.argmax(sarLabel, dim=-1)
                        sentArgmax = torch.argmax(sentLabel, dim=-1)
                        # emoArgmax = torch.argmax(emoLabel, dim=-1)
                        loss1_val = lossFun_val(sar, sarArgmax)
                        loss2_val = lossFun_val(sent, sentArgmax)
                        # loss3_val = lossFun_val(emo, emoArgmax)
                        # loss_val = (loss1_val + loss2_val + loss3_val)/3
                        loss_val = (loss1_val + loss2_val)/3
                        # logger.info('val loss1:%f loss2:%f loss3:%f loss:%f\n'
                        #         % (loss1_val.item(),
                        #             loss2_val.item(),
                        #             loss3_val.item(),
                        #             loss_val.item()))
                        logger.info('val loss1:%f loss2:%f loss:%f\n'
                                % (loss1_val.item(),
                                    loss2_val.item(),
                                    loss_val.item()))

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
                        # pred_emo = np.argmax(
                        #     emo.cpu().detach().numpy(), axis=1)
                        outputsar.append(pred_sar)
                        outputsent.append(pred_sent)
                        # outputemo.append(pred_emo)
                        tarsar.append(label_sar)
                        tarsent.append(label_sent)
                        # taremo.append(label_emo)

                    outputsar = np.concatenate(
                        np.array(outputsar))
                    outputsent = np.concatenate(
                        np.array(outputsent))
                    # outputemo = np.concatenate(
                    #     np.array(outputemo))
                    tarsar = np.concatenate(
                        np.array(tarsar))
                    tarsent = np.concatenate(
                        np.array(tarsent))
                    # taremo = np.concatenate(
                    #     np.array(taremo))

                    
                    sar_f1 = f1_score(
                        tarsar, outputsar, average='micro')
                    sent_f1 = f1_score(
                        tarsent, outputsent, average='micro')
                    # emo_f1 = f1_score(
                    #     taremo, outputemo, average='micro')
                    sar_acc = accuracy_score(
                        tarsar, outputsar)
                    sent_acc = accuracy_score(
                        tarsent, outputsent)
                    # emo_acc = accuracy_score(
                    #     taremo, outputemo)
                    # logger.info(('val-result sar-f1:%f sent-f1:%f emo-f1:%f' +
                    #             ' sar-acc:%f sent-acc:%f emo-acc:%f\n')
                    #             % (sar_f1, sent_f1, emo_f1,
                    #             sar_acc, sent_acc, emo_acc))
                    logger.info(('val-result sar-f1:%f sent-f1:%f' +
                                ' sar-acc:%f sent-acc:%f\n')
                                % (sar_f1, sent_f1,
                                sar_acc, sent_acc))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), stateSavePATH)
        torch.save(model, modelSavePATH)

    
    
  

if __name__ == '__main__':
    logging.basicConfig(filemode='w')
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(
        "result/log/m2modelTextBert_imageAudioFromFile.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info("**********Start print log**********")
    

    trainEval()
    # testModel(modelPATH)

    
