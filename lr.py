#coding:utf-8

#回归

#coding:utf-8

#Linear Regression线性回归

from numpy import *

import matplotlib.pyplot as plt



#导入数据

def loadData(fileName):

    numFeat = len(open(fileName).readline().split('\t')) - 1

    dataMat = []

    labelMat = []

    fr = open(fileName)

    for line in fr.readlines():

        linArr = []

        curline = line.strip().split('\t')#得到每行，并以tab作为间隔

        for i in range(numFeat):

            linArr.append(float(curline[i]))

        dataMat.append(linArr)          

        labelMat.append(float(curline[-1]))

    return dataMat,labelMat



#计算回归系数

def standRegres(x,y):

    xMat = mat(x)

    yMat = mat(y).T

    xTx = xMat.T*xMat

    #采用numpy中的线性代数库linalg，其中linalg.det直接可以计算行列式

    if linalg.det(xTx) == 0.0:

        print "这个行列式是不合法的"

        return

    #求回归系数

    w = xTx.I * (xMat.T * yMat)

    return w

'''

#test

x,y = loadData("ex0.txt")

w = standRegres(x,y)

xMat = mat(x)

yMat = mat(y)

#绘制原数据点

fig = plt.figure()

ax = fig.add_subplot(111)

ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])

    #<matplotlib.collections.CircleCollectin object at 0x04ED9D30>

#plt.show()



#在之前的图像上绘制出拟合直线

xCopy = xMat.copy()

xCopy.sort(0)

yHat = xCopy*w

ax.plot(xCopy[:,1],yHat)

plt.show()



#至此拟合就结束了，那么如何评判拟合的好坏？numpy库提供了相关系数的计算方法，通过命令corrcoef()可以来计算

#预测值和真实值的相关性。

#test

yHat = xMat * w

print corrcoef(yHat.T,yMat)

#采用matlab的cftool工具实现上述功能更简单便捷一点

'''



'''

可以看出相关性还是不错的。但是在某种意义上讲，线性回归是有可能出现欠拟合的现象，因为它求得的是具有最小均方误差的

无偏估计，而模型欠拟合的话势必是不能取得更好的拟合效果，据此可以引入一些偏差，就有了局部加权线性回归。

'''

#局部加权线性回归函数

def lwlr(testPoint,xArr,yArr,k):

    xMat = mat(xArr)

    yMat = mat(yArr).T

    m = shape(xMat)[0]

    weights = mat(eye((m)))

    for i in range(m):

        weights[i,i] = exp((testPoint - xMat[i,:])*(testPoint - xMat[i,:]).T / (-2.0*k**2))

    xTx = xMat.T * (weights * xMat)

    if linalg.det(xTx) == 0:

        print "输入有误"

        return

    ws = xTx.I * xMat.T * weights * yMat

    return testPoint * ws

#为数据点中的每个数据调用lwlr

def lwlrTest(testArr,xArr,yArr,k):

    m = shape(testArr)[0]

    yHat = zeros(m)

    for i in range(m):

        yHat[i] = lwlr(testArr[i],xArr,yArr,k)

    return yHat

'''

#test

x,y = loadData("ex0.txt")

a = lwlr(x[0],x,y,0.002)

b = lwlrTest(x,x,y,0.002)

#采用matplotlib绘制图像

xMat = mat(x)

srtInd = xMat[:,1].argsort(0)#按升序排序，返回下标

xSort = xMat[srtInd][:,0,:]#将xMat按照升序排列

fig = plt.figure()

ax = fig.add_subplot(111)

ax.plot(xSort[:,1],b[srtInd])

ax.scatter(xMat[:,1].flatten().A[0],mat(y).T.flatten().A[0],s = 2,c = 'red')

plt.show()



'''

'''

可以看出采用局部加权回归出来的结果是很理想的，并且可以改变宽度参数观察不同的参数的拟合程度。

但是此种方法的缺点是在对每个点进行预测时都必须遍历整个数据集，这样无疑是增加了工作量。

'''

'''

#结合不同的宽度参数调试结果，计算误差的大小

def rssError(yArr,yHatArr):

    return ((yArr - yHatArr)**2).sum()



#test

x,y = loadData('abalone.txt')

yHat01 = lwlrTest(x[0:99],x[0:99],y[0:99],0.1)

yHat1 = lwlrTest(x[0:99],x[0:99],y[0:99],1)

yHat10 = lwlrTest(x[0:99],x[0:99],y[0:99],10)

rssErro01 = rssError(y[0:99],yHat01.T)

rssErro1 = rssError(y[0:99],yHat1.T)

rssErro10 = rssError(y[0:99],yHat10.T)

print rssErro01

print rssErro1

print rssErro10

'''

def ridgeRegress(xMat,yMat,lam = 0.2):#在没给定lam的时候，默认为0.2

    xTx = xMat.T*xMat

    denom = xTx + eye(shape(xMat)[1])*lam

    if linalg.det(denom) == 0.0:

        print "这个矩阵是错误的，不能求逆"

        return

    ws = denom.I * (xMat.T * yMat)

    return ws

#对数据进行标准化之后，调用30个不同的lam进行计算

def ridgeTest(xArr,yArr):

    xMat = mat(xArr)

    yMat = mat(yArr).T

    yMean = mean(yMat,0)

    yMat = yMat - yMean

    xMeans = mean(xMat,0)

    xVar = var(xMat,0)

    xMat = (xMat - xMeans)/xVar

    numTestPts = 30

    wMat = zeros((numTestPts,shape(xMat)[1]))

    for i in range(numTestPts):

        ws = ridgeRegress(xMat,yMat,exp(i-10))

        wMat[i,:]=ws.T

    return wMat

'''

#test

x,y = loadData('abalone.txt')



ridgeWeights = ridgeTest(x,y)



#print ridgeWeights

#这样就得到了30个不同lam所对应的回归系数，为了看到缩减的效果，可采用matplotlib绘图

fig = plt.figure()

ax = fig.add_subplot(111)

ax.plot(ridgeWeights)

plt.show()

'''

from time import sleep

import json

import urllib2

def searchForSet(retX, retY, setNum, yr, numPce, origPrc):

    sleep(10)

    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'

    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)

    pg = urllib2.urlopen(searchURL)

    retDict = json.loads(pg.read())

    for i in range(len(retDict['items'])):

        try:

            currItem = retDict['items'][i]

            if currItem['product']['condition'] == 'new':

                newFlag = 1

            else: newFlag = 0

            listOfInv = currItem['product']['inventories']

            for item in listOfInv:

                sellingPrice = item['price']

                if  sellingPrice > origPrc * 0.5:

                    print "%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice)

                    retX.append([yr, numPce, newFlag, origPrc])

                    retY.append(sellingPrice)

        except: print 'problem with item %d' % i

    

def setDataCollect(retX, retY):

    searchForSet(retX, retY, 8288, 2006, 800, 49.99)

    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)

    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)

    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)

    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)

    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)



lgx = []

lgy = []

x = setDataCollect(lgx,lgy)

print x



































































































        
