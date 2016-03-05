#coding:utf-8

'''
贝叶斯的python实现
分类的原理是通过某对象的先验概率，利用贝叶斯公式计算出它的后验概率（对象属于某一类的概率），选取具有最大后验概率的类作为该对象所属的类。
'''

from numpy import *

##从文本中构建向量
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    ##分别表示标签
    return postingList,classVec ##返回输入数据和标签向量
                 
def createVocabList(dataSet):
    vocabSet = set([])  
    for document in dataSet:
        vocabSet = vocabSet | set(document) 
    return list(vocabSet)##输出不重复的元素

def setOfWords2Vec(vocabList, inputSet):###判断了一个词是否出现在一个文档当中。
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print "the word: %s is not in my Vocabulary!" % word
    return returnVec###输入中的元素在词汇表时，词汇表相应位置为1，否则为0

'''
#测试
dataSet,classes = loadDataSet()
print(dataSet)
vocabList = createVocabList(dataSet)
print(vocabList)
setWordsVec = setOfWords2Vec(vocabList,dataSet[0])
print(setWordsVec)
'''
##得到每个特征的条件概率

def trainNB0(trainMatrix,trainCategory):###输入的文档信息和标签

    numTrainDocs = len(trainMatrix)

    numWords = len(trainMatrix[0])

    pAbusive = sum(trainCategory)/float(numTrainDocs)

    p0Num = ones(numWords)

    p1Num = ones(numWords)      

    p0Denom = 2.0

    p1Denom = 2.0                     

    for i in range(numTrainDocs):

        if trainCategory[i] == 1:

            p1Num += trainMatrix[i]

            p1Denom += sum(trainMatrix[i])

        else:

            p0Num += trainMatrix[i]

            p0Denom += sum(trainMatrix[i])

    p1Vect = log(p1Num/p1Denom)          

    p0Vect = log(p0Num/p0Denom)   

    return p0Vect,p1Vect,pAbusive

'''

#测试

dataSet,classes = loadDataSet()

vocabList = createVocabList(dataSet)

trainMat = []

for item in dataSet:

    trainMat.append(setOfWords2Vec(vocabList,item))

                    

p0v,p1v,pAb = trainNB0(trainMat,classes)

print(p0v)

print(p1v)

print(pAb)

'''

#分类

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):

    p1 = sum(vec2Classify * p1Vec) + log(pClass1)

    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)

    if p1 > p0:

        return 1

    else:

        return 0



#词袋模型(返回所有词汇出现的次数）

def bagOfWords2VecMN(vocabList, inputSet):

    returnVec = [0]*len(vocabList)

    for word in inputSet:

        if word in vocabList:

            returnVec[vocabList.index(word)] += 1

    return returnVec



def testingNB():

    listOPosts,listClasses = loadDataSet()

    myVocabList = createVocabList(listOPosts)

    trainMat=[]

    for postinDoc in listOPosts:

        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))

    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))

    testEntry = ['love', 'my', 'dalmation']

    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))

    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)

    testEntry = ['stupid', 'garbage']

    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))

    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)

'''

#测试

testingNB()

'''



###根据分类器过滤垃圾邮件

#单个输出字母个数大于2的单词

def textParse(bigString):    #input is big string, #output is word list

    import re

    listOfTokens = re.split(r'\W*', bigString)

    return [tok.lower() for tok in listOfTokens if len(tok) > 2] 

    

def spamTest():

    docList=[]; classList = []; fullText =[]

    for i in range(1,26):

        wordList = textParse(open('spam/%d.txt' % i).read())

        docList.append(wordList)

        fullText.extend(wordList)

        classList.append(1)

        wordList = textParse(open('ham/%d.txt' % i).read())

        docList.append(wordList)

        fullText.extend(wordList)

        classList.append(0)

    vocabList = createVocabList(docList)

    trainingSet = range(50); testSet=[]

    for i in range(10):

        randIndex = int(random.uniform(0,len(trainingSet)))

        testSet.append(trainingSet[randIndex])

        del(trainingSet[randIndex])  

    trainMat=[]; trainClasses = []

    for docIndex in trainingSet:

        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))

        trainClasses.append(classList[docIndex])

    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))

    errorCount = 0

    for docIndex in testSet:        

        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])

        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:

            errorCount += 1

            print "错误",docList[docIndex]

    print "错误率为：",float(errorCount / len(testSet))



#测试

spamTest()

###会随机输出在10封邮件上的分类错误率。
