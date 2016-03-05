#coding:utf-8

'''
决策树的实现
决策树是一种预测模型，分类算法，可以利用决策树表示出数据内部蕴含的知识。
'''

from math import log
import operator
import matplotlib.pyplot as plt

###计算香农熵(为float类型）
def calShang(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}##创建字典    
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0    
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

def creatDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels
'''
#test
myData,labels = creatDataSet()
print("原数据为：",myData)
print("标签为：",labels)
shang = calShang(myData)
print("香农熵为：",shang)
'''

###划分数据集（以指定特征将数据进行划分）
def splitDataSet(dataSet,feature,value):##传入待划分的数据集、划分数据集的特征以及需要返回的特征的值
    newDataSet = []
    for featVec in dataSet:
        if featVec[feature] == value:
            reducedFeatVec = featVec[:feature]
            reducedFeatVec.extend(featVec[feature + 1:])
            newDataSet.append(reducedFeatVec)
    return newDataSet

'''
#测试
myData,labels = creatDataSet()
print("原数据为：",myData)
print("标签为：",labels)
split = splitDataSet(myData,0,1)
print("划分后的结果为:",split)
'''

##选择最好的划分方式(选取每个特征划分数据集，从中选取信息增益最大的作为最优划分)在这里体现了信息增益的概念
def chooseBest(dataSet):
    featNum = len(dataSet[0]) - 1
    baseEntropy = calShang(dataSet)
    bestInforGain = 0.0
    bestFeat = -1##表示最好划分特征的下标

    for i in range(featNum):
        featList = [example[i] for example in dataSet] #列表
        uniqueFeat = set(featList)##得到每个特征中所含的不同元素
        newEntropy = 0.0
        for value in uniqueFeat:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet) / len(dataSet)
            newEntropy += prob * calShang(subDataSet)
        inforGain = baseEntropy - newEntropy
        if (inforGain > bestInforGain):
            bestInforGain = inforGain
            bestFeature = i#第i个特征是最有利于划分的特征
    return bestFeature
    
'''
##测试
myData,labels = creatDataSet()
best = chooseBest(myData)
print(best)
'''

#返回出现次数最多的分类名称
def majorClass(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    #降序排序，可以指定reverse = true
    sortedClassCount = sorted(classcount.iteritems(),key = operator.itemgetter(1),reverse = true)
    return sortedClassCount[0][0]

#创建树
def creatTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]         
    if len(dataSet[0]) == 1:        
        return majorClass(classList)
    bestFeat = chooseBest(dataSet) 
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = creatTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

'''
#测试
myData,labels = creatDataSet()
mytree = creatTree(myData,labels)
print(mytree)
'''

##采用matplotlib绘制树形图
ecisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

 #绘制树节点
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )

##获取节点的数目和树的层数 
def getNumLeafs(myTree):
numLeafs = 0
#firstStr = myTree.keys()[0]
firstSides = list(myTree.keys())
firstStr = firstSides[0]#找到输入的第一个元素
secondDict = myTree[firstStr]
for key in secondDict.keys():
if type(secondDict[key]) == dict:
numLeafs += getNumLeafs(secondDict[key])
else: numLeafs += 1
return numLeafs

def getTreeDepth(myTree):
maxDepth = 1
firstSides = list(myTree.keys())
firstStr = firstSides[0]#找到输入的第一个元素
#firstStr = myTree.keys()[0]
secondDict = myTree[firstStr]
for key in secondDict.keys():
if type(secondDict[key]) == dict:
thisDepth = 1 + getTreeDepth(secondDict[key])
else: thisDepth = 1
if thisDepth > maxDepth: maxDepth = thisDepth
return maxDepth

def retrieveTree(i):
listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
{'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
]
return listOfTrees[i]


#测试
mytree = retrieveTree(0)
print(getNumLeafs(mytree))
print(getTreeDepth(mytree))

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )
    
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)  
    depth = getTreeDepth(myTree)
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]#找到输入的第一个元素
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':   
            plotTree(secondDict[key],cntrPt,str(key))        
        else:   
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plt.show()
    

#测试
mytree = retrieveTree(0)
print(mytree)
createPlot(mytree)
