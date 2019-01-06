from numpy import *

import operator

from os import listdir

def creatDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels =['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0] #shape[0]为读取行数，shape[1]为读取列数
    diffMat = tile(inX,(dataSetSize,1))-dataSet #tile:将inX复制成(dataSetSize行，1列)
    sqDiffMat=diffMat**2
    sqDistance = sqDiffMat.sum(axis=1) #axis =1表示矩阵的每一行元素相加
    distance = sqDistance**0.5 #开根号
    sortedDistIndicies = distance.argsort() #返回从小到大的顺序值
    classCount={} #建立一个字典，用于计数
    for i in range(k):#按顺序对标签进行计数
        voteIlabel = labels[sortedDistIndicies[i]] #按之前排序值依次对标签计数
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1  #对字典进行抓取，此时字典是空的，把value作为key，value是label出现的次数
        #但是数组是从0开始，计数是从1开始，所以后面要加1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)#返回一个列表按照第二个元素降序排列
    return sortedClassCount[0][0]


def file2matrix(filename):
    love_dictionary = {'largeDoses': 3, 'smallDoses': 2, 'didntLike': 1}
    fr = open(filename)
    arrayOLines = fr.readlines() #按行读取并存在arrayOLines里
    numberOfLines = len(arrayOLines) #读取其行数
    returnMat = zeros((numberOfLines,3)) #numberOfLines行,3列
    classLabelVector = [] #建立一个单列矩阵，存储其类
    index = 0 #索引值先清零
    for line in arrayOLines:
        line = line.strip() #将每一行的回车去掉
        listFromLine = line.split('\t') #每个制表符分隔，返回分隔后的字符串
        returnMat[index,:] = listFromLine[0:3] #将每一行的前三个元素依次赋值给之前预留的zero矩阵
       # classLabelVector.append(int(listFromLine[-1]))#将每行的最后一列赋值给单列矩阵
        if (listFromLine[-1].isdigit()):
            classLabelVector.append(int(listFromLine[-1]))
        else:
            classLabelVector.append(love_dictionary.get(listFromLine[-1]))
        index += 1 #进行下一次循环
    return returnMat,classLabelVector #返回两个矩阵，一个是三个特征组成的特征矩阵，一个是类矩阵

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals-minVals
    normDataSet = zeros(shape(dataSet))
    m= dataSet.shape[0]
    normDataSet = dataSet-tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

def datingClassTest():
    filename = "datingTestSet.txt"
    hoRation = 0.10
    datingDataMat,datingLabels =file2matrix('datingTestSet.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m=normMat.shape[0]
    numTestVecs = int(m*hoRation)
    errorCount =0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print('The classifier came back with :%d,the real answer is : %d'%(classifierResult,datingLabels[i]))
        if(classifierResult !=datingLabels[i]):
            errorCount +=1
    print('The total error rate is  %.2f %%'%(errorCount/float(numTestVecs)*100))

def classifyPerson():
    resultList= ['not at all','in small doses','in large doses']
    precentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier mile earned one year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels = file2matrix("datingTestSet2.txt")
    normMat,ranges,minVals=autoNorm(datingDataMat)
    inArr = array([ffMiles,precentTats,iceCream])
    classifyResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print('Your will probably like this person:',resultList[classifyResult-1])

def imag2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline() #按行读取
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwlabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        #fileStr = fileNameStr.split('.') [0] #分离，取出第一个元素，为0 #相当于去除了文件拓展后缀名
        classNumStr = int(fileNameStr.split('_')[0]) #分离，取出第一个元素，获取了图像类别标签
        hwlabels.append(classNumStr)
        trainingMat[i,:] = imag2vector('trainingDigits/%s'%fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNumStr = int(fileNameStr.split('_')[0])
        verctorUnderTest = imag2vector('testDigits/%s' %fileNameStr)
        classifierResult=classify0(verctorUnderTest,trainingMat,hwlabels,1)
        print('the classifier came back with :%d,the real answer is %d'%(classifierResult,classNumStr))
        if(classifierResult!= classNumStr):
            errorCount +=1
    print('\n the total number of error is :%d'%errorCount)
    print('\n the total error rate is %f %%' %(errorCount/float(mTest)*100))

handwritingClassTest()













