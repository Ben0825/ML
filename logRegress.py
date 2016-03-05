#coding:utf-8

from numpy import *

import math

import matplotlib.pyplot as plt



#导入数据

def loadDataSet():

  dataMat = []

  labelMat = []

  fr = open('testSet.txt')

  for line in fr.readlines():

    lineArr = line.strip().split()#将文本中的每行中的字符一个个分开，变成list

    dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])

    labelMat.append(int(lineArr[2]))

  return dataMat,labelMat



#定义sigmoid函数

def sigmoid(inX):

    return 1.0/(1+exp(-inX))



#梯度上升方法求出回归系数

def gradAscent(data,label):

  dataMat = mat(data)

  labelMat = mat(label).transpose()

  m,n = shape(dataMat)

  alpha = 0.001

  maxCycles = 500

  weights = ones((n,1))

  for item in range(maxCycles):

    h = sigmoid(dataMat * weights)

    error = (labelMat - h)#注意labelMat中的元素的数据类型应为int

    weights = weights + alpha * dataMat.transpose() * error

  return weights



'''

#测试

data,label = loadDataSet()

print gradAscent(data,label)

'''



##求出回归系数之后，就确定了不同数据类别之间的分隔线，为了便于理解，可以画出那条线

def plotBestFit(weights):

  dataMat,labelMat = loadDataSet()

  dataArr = array(dataMat)

  n = shape(dataArr)[0]

  xcode1 = []

  ycode1 = []

  xcode2 = []

  ycode2 = []

  for i in range(n):

    if int(labelMat[i]) == 1:

      xcode1.append(dataArr[i,1])

      ycode1.append(dataArr[i,2])

    else:

      xcode2.append(dataArr[i,1])

      ycode2.append(dataArr[i,2])

  fig = plt.figure()

  ax = fig.add_subplot(111)

  ax.scatter(xcode1,ycode1,s = 30,c = 'red',marker = 's')

  ax.scatter(xcode2,ycode2,s = 30,c = 'green')

  x = arange(-3.0,3.0,0.1)

  y = (-weights[0] - weights[1] * x) / weights[2]

  ax.plot(x,y)

  plt.xlabel('x1')

  plt.ylabel('y1')

  plt.show()

  

'''

#测试

data,label = loadDataSet()

weights = gradAscent(data,label)

plotBestFit(weights.getA())

'''

###根据上图中显示的分类结果，有四个点是分错的，所以分类是相当不准确的，故需对算法进一步改进



#梯度上升算法在每次更新回归系数的时候都要遍历整个数据集，当数据增大时，相应的算法的时间复杂度就会增大

#所以可以一次仅用一个样本点来更新回归系数，称为随机梯度上升算法，

#所有回归系数初始化为1

#对数据集中的每个样本

# 计算样本的梯度

# 使用alpha * gradient更新回归系数值

#返回回归系数值



def stocGradAscent(dataMatrix, classLabels):

    m,n = shape(dataMatrix)

    alpha = 0.01

    weights = ones(n) 

    for i in range(m):

        h = sigmoid(sum(dataMatrix[i]*weights))

        error = classLabels[i] - h

        weights = weights + alpha * error * dataMatrix[i]

    return weights

'''

#测试

data,label = loadDataSet()

weights = stocGradAscent(array(data),label)

plotBestFit(weights)

'''

####采用这种分类出来的结果大概有三分之一都分错了

#鉴于判断算法性能优劣的可靠方法是看算法是否收敛，也就是说参数是否达到了稳定值，故对上述算法做出修改



def stocGradAscent1(dataMatrix, classLabels, numIter=150):

    m,n = shape(dataMatrix)

    weights = ones(n)   #initialize to all ones

    for j in range(numIter):

        dataIndex = range(m)

        for i in range(m):

            alpha = 4/(1.0+j+i)+0.0001    

            randIndex = int(random.uniform(0,len(dataIndex)))

            h = sigmoid(sum(dataMatrix[randIndex]*weights))

            error = classLabels[randIndex] - h

            weights = weights + alpha * error * dataMatrix[randIndex]

            del(dataIndex[randIndex])

    return weights

''' 

#测试

data,label = loadDataSet()

weights = stocGradAscent1(array(data),label)

plotBestFit(weights)

'''



'''

####采用Logistic回归患有疝病的马的存活问题

##多给的数据中有一部分是缺失的，首先得去处理缺失值

因为我们得数据都是从机器上收集而来的，机器上的某个传感器啊什么的要是损坏了有时会导致特征无效，而有时候数据是相当昂贵的，

扔掉和重新获取都是不可取的，在这种情况下，必须采取一些办法来解决：

1. 使用可用特征的均值来填补缺失值；

2. 使用特殊值来填补缺失值，比如-1；

3. 忽略有缺失值的样本；

4. 使用相似样本的均值填补缺失值；

5. 使用另外的机器学习算法预测缺失值

我们对本节要使用的数据进行预处理，使其可以顺利的使用分类算法。在这之前我们必须做两件事：一是所有的缺失值使用一个实数值进行替换；

如果一条数据的类别标签已经丢失，那么最简单的做法就是将其丢弃。

'''





#测试算法，采用Logistic回归进行分类

#只需要将测试集的各个特征向量乘以最优化得来的回归系数，再将该乘积的结果求和，

#最后输入到sigmoid函数中即可

def classifyVector(inX,weights):

  prob = sigmoid(sum(inX * weights))

  if prob > 0.5:

    return 1.0

  else:

    return 0.0

def colicTest():

    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')

    trainingSet = []; trainingLabels = []

    for line in frTrain.readlines():

        currLine = line.strip().split('\t')

        lineArr =[]

        for i in range(21):

            lineArr.append(float(currLine[i]))

        trainingSet.append(lineArr)

        trainingLabels.append(float(currLine[21]))

    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)

    errorCount = 0; numTestVec = 0.0

    for line in frTest.readlines():

        numTestVec += 1.0

        currLine = line.strip().split('\t')

        lineArr =[]

        for i in range(21):

            lineArr.append(float(currLine[i]))

        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):

            errorCount += 1

    errorRate = (float(errorCount)/numTestVec)

    print "错误率是：",errorRate

    return errorRate

def multiTest():

    numTests = 10; errorSum=0.0

    for k in range(numTests):

        errorSum += colicTest()

    print "平均错误率是",(numTests, errorSum/float(numTests))

        

multiTest()



































  





























  

























  
