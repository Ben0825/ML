# -*- coding: cp936 -*-

'''

Apriori算法

Ben

2015.09.28

'''

#coding:utf-8

from numpy import *



def loadData():

    return[[1,3,4],[2,3,5],[1,2,3,5],[2,5]]



def createC1(dataSet):

    c1 = []

    for transaction in dataSet:

        for item in transaction:

            if not [item] in c1:

                c1.append([item])

    c1.sort()

    return map(frozenset,c1)



#构建频繁项集

def scanD(D,Ck,minSupport):

    ssCnt = {}

    for tid in D:

        for can in Ck:

            if can.issubset(tid):#判断tid是否在can中

                if not ssCnt.has_key(can):

                    ssCnt[can] = 1

                else:

                    ssCnt[can] += 1

    numItems = float(len(D))

    retList = []

    supportData = {}

    for key in ssCnt:

        support = ssCnt[key] / numItems

        if support >= minSupport:

            retList.insert(0,key)

        supportData[key] = support

    return retList,supportData

'''

#test

dataSet = loadData()

c1 = createC1(dataSet)

D = map(set,dataSet)

L1,supportData = scanD(D,c1,0.5)

print L1

print supportData

'''



#构建多个商品对应的项集

def aprioriGen(Lk,k):

    retList = []

    lenLk = len(Lk)

    for i in range(lenLk):

        for j in range(i+1,lenLk):

            L1 = list(Lk[i])[:k-2]

            L2 = list(Lk[j])[:k-2]

            L1.sort()

            L2.sort()

            if L1 == L2:

                retList.append(Lk[i]|Lk[j])

    return retList



def apriori(dataSet,minSupport = 0.5):

    C1 = createC1(dataSet)

    D = map(set,dataSet)

    L1,supportData = scanD(D,C1,minSupport)

    L = [L1]

    k = 2

    while (len(L[k-2]) > 0):

        Ck = aprioriGen(L[k-2],k)

        Lk,supK = scanD(D,Ck,minSupport)

        supportData.update(supK)

        L.append(Lk)

        k += 1

    return L,supportData

'''

#test

dataSet = loadData()

minSupport = 0.5

a,b = apriori(dataSet,minSupport)

print a

print b

'''



#使用关联规则生成函数

def generateRules(L,supportData,minConf = 0.7):

    bigRuleList = []

    for i in range(1,len(L)):

        for freqSet in L[i]:

            H1 = [frozenset([item]) for item in freqSet]

            if (i > 1):

                rulesFromConseq(freqSet,H1,supportData,bigRuleList,minConf)

            else:

                calcConf(freqSet,H1,supportData,bigRuleList,minConf)

    return bigRuleList



#集合右边一个元素

def calcConf(freqSet,H,supportData,brl,minConf = 0.7):

    prunedH = []

    for conseq in H:

        conf = supportData[freqSet]/supportData[freqSet - conseq]

        if conf >= minConf:

            print freqSet - conseq,'-->',conseq,'conf:',conf

            brl.append((freqSet-conseq,conseq,conf))

            prunedH.append(conseq)

    return prunedH



#生成更多的关联规则

def rulesFromConseq(freqSet,H,supportData,br1,minConf = 0.7):

    m = len(H[0])

    if (len(freqSet)>(m + 1)):

        Hmp1 = aprioriGen(H,m+1)

        Hmp1 = calcConf(freqSet,Hmp1,supportData,br1,minConf)

        if (len(Hmp1) > 1):

            rulesFromConseq(freqSet,Hmp1,supportData,br1,minConf)

'''

#test

dataSet = loadData()

minSupport = 0.5

L,suppData = apriori(dataSet,minSupport)

rules = generateRules(L,suppData,minConf = 0.5)

print rules

'''



#test

mushDatSet = [line.split() for line in open('mushroom.dat').readlines()]

L,supportData = apriori(mushDatSet,minSupport = 0.3)

for item in L[1]:

    if item.intersection('2'):

        print item

        





























                   

       







                

   


