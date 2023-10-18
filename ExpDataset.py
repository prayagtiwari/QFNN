import torch
from torch.utils.data import Dataset
from transformers import AlbertTokenizer, AlbertConfig
import numpy as np
import csv

import math


def readcsv(fileName):

    with open(fileName, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        uttNameList = []
        for i in reader:
            if i[0] != '':
                uttNameList.append(i[0])
        uttNameList = list(set(uttNameList))
        uttNameList.remove("KEY")
        uttDict = {}
        for name in uttNameList:
            uttDict[name] = {}
            uttDict[name]['utterance'] = []
            uttDict[name]['sarcasm-label'] = ''
            uttDict[name]['sentiment-label'] = ''
            uttDict[name]['emotion-label'] = ''
            uttDict[name]['utt-number'] = ''

    with open(fileName, 'r', encoding='utf-8') as f1:
        reader = csv.reader(f1)
        for item in reader:
            if item[0] == 'KEY' or item[0] == '':
                continue
            uttDict[item[0]]['sarcasm-label'] = item[4]
            uttDict[item[0]]['sentiment-label'] = item[5]
            uttDict[item[0]]['emotion-label'] = item[7]
            uttDict[item[0]]['utterance'].append(item[2])
            uttDict[item[0]]['utt-number'] = item[0]
    return uttDict, uttNameList



def  processUttDict(uttDict):

    for key in uttDict.keys():
        lenUtt = len(uttDict[key]['utterance'])
        if lenUtt == 2:
            uttDict[key]['utterance'].insert(0, '')
            uttDict[key]['utterance'].insert(0, '')
        elif lenUtt > 4:
            uttDict[key]['utterance'] = uttDict[key]['utterance'][-4:]
        elif lenUtt == 3:
            uttDict[key]['utterance'].insert(0, '')
        else:
            continue

    return uttDict


class MustardDataset(Dataset):

    def __init__(self, datatye):
        super().__init__()

        self.datatye = datatye
        if self.datatye == 'train':
            datafile = 'data/mustard-dataset-train.csv'
        elif self.datatye == 'dev':
            datafile = 'data/mustard-dataset-dev.csv'
        elif self.datatye == 'test':
            datafile = 'data/mustard-dataset-test.csv'
        uttDict, self.uttNameList = readcsv(datafile)
        uttDict = processUttDict(uttDict)
        self.uttList = list(uttDict.values())

        self.frameNumbers = 4

    def __getitem__(self, index):
        uttName = self.uttList[index]['utt-number']

        textPath = \
            "token/mustard/txt_final/"+uttName+"/"
        textFea = [np.loadtxt(
                        textPath+str(num)+'.txt',
                        dtype=float) for num in range(4)]
        textFea = [np.expand_dims(item, axis=0) for item in textFea]
        textFea = np.concatenate(textFea, axis=0)
        textFea = torch.tensor(textFea)

        imagePath = "token/mustard/img/"+uttName+"/"
        imageFea = [np.loadtxt(
                        imagePath+str(num)+'.txt',
                        dtype=float) for num in range(4)]
        imageFea = [item.reshape(24,56,56) for item in imageFea]
        imageFea = [np.expand_dims(item, axis=0) for item in imageFea]
        imageFea = np.concatenate(imageFea, axis=0)
        imageFea = torch.tensor(imageFea)


        audioPath = "token/mustard/audio_final/"+uttName+"/"
        auFea = [np.loadtxt(
                        audioPath+str(num)+'.txt',
                        dtype=float) for num in range(4)]
        auFea = [np.expand_dims(item, axis=0) for item in auFea]
        auFea = np.concatenate(auFea, axis=0)
        auFea = torch.tensor(auFea)

        sarcasmStr = self.uttList[index]['sarcasm-label']
        sentimentStr = self.uttList[index]['sentiment-label']
        emotionStr = self.uttList[index]['emotion-label']
        if 'True' == sarcasmStr:
            sarcasmLabel = np.array([0, 1], dtype=np.int8)
        else:
            sarcasmLabel = np.array([1, 0], dtype=np.int8)

        sentimentLabel = np.zeros(3, dtype=np.int8)
        if -1 == int(sentimentStr):
            sentimentLabel[0] = 1
        elif 0 == int(sentimentStr):
            sentimentLabel[1] = 1
        else:
            sentimentLabel[2] = 1
        emotionLabel = np.zeros(9, dtype=np.int8)
        emotionLabel[int(emotionStr.split(',')[0])-1] = 1

        return [textFea, imageFea,
                auFea],\
            [sarcasmLabel, sentimentLabel, emotionLabel]

    def __len__(self):
        return len(self.uttList)


class MemotionDataset(Dataset):

    def __init__(self):
        super().__init__()
        self.textDataset = 'memotion-dataset.csv'
        uttDict = {}
        with open(self.textDataset, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            a = 0
            for line in reader:
                if line[0] == 'number':
                    continue
                if line[1].split('.')[0] in self.imageNameList:
                    uttDict[line[0]] = {}
                    uttDict[line[0]]['utterence'] = line[3]
                    uttDict[line[0]]['imageName'] = line[1]
                    uttDict[line[0]]['sarcasm-label'] = line[5]
                    uttDict[line[0]]['sentiment-label'] = line[8]
        self.uttDict = uttDict

    def __getitem__(self, index):
        
        text = self.uttDict[index]['utterence']
        return 1
