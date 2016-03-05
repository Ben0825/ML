#coding:utf-8

'''
create on Mar 5,2016
KNN算法的实现
@author:Ben
KNN是通过测量不同特征值之间的距离进行分类，在训练集中数据和标签已知的情况下，输入测试数据，将测试数据的特征与训练集中对应的特征进行相互比较，找到训练集中与之最为相似的前K个数据，则该测试数据对应的类别就是K个数据中出现次数最多的那个分类
'''

from numpy import *
import operator

##给出训练数据以及对应的类别
def createDataSet():
    group = array([[1.0,2.0],[1.2,0.1],[0.1,1.4],[0.3,3.5]])
    labels = ['A','A','B','B']
    return group,labels

###通过KNN进行分类
def classify(input,dataSet,label,k):
    dataSize = dataSet.shape[0]
    ####计算欧式距离
    diff = tile(input,(dataSize,1)) - dataSet
    sqdiff = diff ** 2
    squareDist = sum(sqdiff,axis = 1)###行向量分别相加，从而得到新的一个行向量
    dist = squareDist ** 0.5
    
    ##对距离进行排序
    sortedDistIndex = argsort(dist)##argsort()根据元素的值从大到小对元素进行排序，返回下标

    classCount={}
    for i in range(k):
        voteLabel = label[sortedDistIndex[i]]
        ###对选取的K个样本所属的类别个数进行统计
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1
    ###选取出现的类别次数最多的类别
    maxCount = 0
    for key,value in classCount.items():
        if value > maxCount:
            maxCount = value
            classes = key

    return classes    

#test
dataSet,labels = createDataSet()
input = array([1.1,0.3])
K = 3
output = classify(input,dataSet,labels,K)
print("测试数据为:",input,"分类结果为：",output)


