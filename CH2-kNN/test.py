import kNN
import matplotlib
import matplotlib.pyplot as plt
from numpy import *

from os import listdir

group,labels = kNN.creatDataSet()
#
# print(group)
# print(labels)
#print(kNN.classsify0([0,0],group,labels,3))
datingDataMat,datingLabels= kNN.file2matrix('datingTestSet.txt')
datingDateMat,ranges,minVals = kNN.autoNorm(datingDataMat)
# print(datingDataMat)
# print(datingLabels)
print(ranges)
print(minVals)
fig = plt.figure()
ax = fig.add_subplot(121)
ax1 = fig.add_subplot(122)
ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))

ax1.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*array(datingLabels),15.0*array(datingLabels))


plt.xlabel('玩游戏视频所消耗的时间比')
plt.ylabel('每周消费的冰淇淋公斤数')
plt.legend('ax1')
# plt.title('scatter')
# plt.legend('1')

# ax.xlabel('x')
# ax.ylabel('y')
# ax.title('scatter')
# ax.legend('1')
plt.show()
