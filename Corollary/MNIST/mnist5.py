# 使用sklearn的函数来获取MNIST数据集
from sklearn.datasets import fetch_openml
import numpy as np
import os
import psutil

# to make this notebook's output stable across runs
np.random.seed(4242)

import time
time_begin=time.time()
mnist=fetch_openml('mnist_784')
time_end=time.time()
time_data=time_end-time_begin
print('加载数据集所用时间:',time_data)

X,y=mnist['data'],mnist['target']
print(X.shape) #数据X共有7万张图片，每张图片有784个特征。因为图片是28×28像素，每个特征代表了一个像素点的强度，从0（白色）到255（黑色），

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=8888)

#基分类器1
import time
import os

from sklearn.tree import DecisionTreeClassifier
memory_start1=psutil.Process(os.getpid()).memory_info().rss/1024/1024 #MB
start_time_base1=time.time()
classifier1=DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=25,min_samples_split=5,random_state=8842)#第一次测试最优
classifier1.fit(X_train,y_train)
score1=classifier1.score(X_test,y_test)
print("基分类器1score:",score1)
end_time_base1=time.time()
base_time1=end_time_base1-start_time_base1
print('基分类器1(25)所用时间:',base_time1)
memory_end1 =psutil.Process(os.getpid()).memory_info().rss/1024/1024  #MB
memory_usage1=memory_end1-memory_start1
print('基分类器1(25)所用内存(MB):',memory_usage1) #进程占用的物理内存大小

import joblib

#保存Model
joblib.dump(classifier1,'model_mnist5/base1_learner_mnist.pkl')
print("基分类器1pkl_size(MB):",os.path.getsize('model_mnist5/base1_learner_mnist.pkl')/1024/1024) #MB

# Bagging1
def get_dir_size(target_dir):
    pkl_size=[] #MB
    dir_list=os.listdir(target_dir)
    print(dir_list)
    #计算每个文件的大小
    for file in dir_list:
        file = os.path.join(target_dir, file)
        #如果是文件，直接通过getsize计算大小并加到size中
        if os.path.isfile(file):
            pkl_size.append(os.path.getsize(file)/1024/1024) #MB
    return pkl_size

from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def bagging_clf(base_learner,target_dir,n):
    #性能指标
    time_bagging=[] #s
    top1_bagging=[]
    m_percent_bagging=[]

    for i in tqdm(n):
        # cpu_start = psutil.Process(os.getpid()).cpu_percent(interval=20) #bagging interval=20/boosting interval=150 不测这个变量，因为由于函数调用会阻塞（1s无法准确反映cpu实际情况）/20/150 秒，因此可能会对time测量产生一定影响。
        memory_start=psutil.Process(os.getpid()).memory_info().rss/1024/1024 #MB
        start_time_bagging=time.time()

        # 创建Bagging集成学习器
        bagging_clf = BaggingClassifier(base_estimator=base_learner, n_estimators=i, random_state=8842,n_jobs=-1,bootstrap=True)
        bagging_clf.fit(X_train, y_train)
        y_pred_bagging = bagging_clf.predict(X_test)

        # 评估性能
    #     print("Bagging Accuracy:", accuracy_score(y_test, y_pred_bagging))
        top1_bagging.append(accuracy_score(y_test, y_pred_bagging))

        end_time_bagging=time.time()
        bagging_time=end_time_bagging-start_time_bagging
    #     print('Bagging所用时间：',bagging_time)
        time_bagging.append(bagging_time)
        memory_end  =psutil.Process(os.getpid()).memory_info().rss/1024/1024  #MB
        m_percent_end_bagging= memory_end - memory_start #memory_usage 
        m_percent_bagging.append(m_percent_end_bagging)

        joblib.dump(bagging_clf,f'{target_dir}/bagging_mnist_{i}.pkl')
    
    return time_bagging,top1_bagging,m_percent_bagging

n= [20,40,60,80,100,120,140,160,180,200,300]

time_bagging1,top1_bagging1,m_percent_bagging1=bagging_clf(classifier1,'model_mnist5/model1_bagging',n)
pkl_size_bagging1=get_dir_size('model_mnist5/model1_bagging')

#保存结果
import pandas as pd
c1={"基分类器个数" : n,
   "performance" : top1_bagging1,
  "时间":time_bagging1,
  "内存使用率":m_percent_bagging1,
  "pkl_size_bagging1":pkl_size_bagging1}#将列表a，b转换成字典
bagging_result1=pd.DataFrame(c1)#将字典转换成为数据框
print("mnist_base1_bagging:",bagging_result1)
bagging_result1.to_csv('model_mnist5/bagging_MNIST_result1.csv')

# Boosting1
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def boosting_clf(base_learner,target_dir,n):

    #性能指标
    time_boosting=[] #s
    top1_boosting=[]
    m_percent_boosting=[]
    

    for i in tqdm(n):
        memory_start=psutil.Process(os.getpid()).memory_info().rss/1024/1024 #MB
        start_time_boosting=time.time()
        
        # 创建Bagging集成学习器
        boosting_clf = AdaBoostClassifier(base_estimator=base_learner, n_estimators=i, random_state=8842)
        boosting_clf.fit(X_train, y_train)
        y_pred_boosting = boosting_clf.predict(X_test)

        # 评估性能
        top1_boosting.append(accuracy_score(y_test, y_pred_boosting))

        end_time_boosting=time.time()
        boosting_time=end_time_boosting-start_time_boosting
        time_boosting.append(boosting_time)
        memory_end  =psutil.Process(os.getpid()).memory_info().rss/1024/1024  #MB
        m_percent_end_boosting= memory_end - memory_start #memory_usage
        m_percent_boosting.append(m_percent_end_boosting)
 
        joblib.dump(boosting_clf,f'{target_dir}/boosting_mnist_{i}.pkl')    
    
    return time_boosting,top1_boosting,m_percent_boosting

time_boosting1,top1_boosting1,m_percent_boosting1=boosting_clf(classifier1,'model_mnist5/model1_boosting',n)
pkl_size_boosting1=get_dir_size('model_mnist5/model1_boosting')
#保存结果
import pandas as pd
cb1={"基分类器个数" : n,
   "performance" : top1_boosting1,
  "时间":time_boosting1,
  "内存使用率":m_percent_boosting1,
  "pkl_size_boosting1":pkl_size_boosting1}#将列表a，b转换成字典
boost_result1=pd.DataFrame(cb1)#将字典转换成为数据框
print("mnist_base1_boosting:",boost_result1)
boost_result1.to_csv('model_mnist5/boosting_MNIST_result1.csv')


#基分类器2
memory_start2=psutil.Process(os.getpid()).memory_info().rss//1024/1024 #MB
start_time_base2=time.time()
classifier2=DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=15,min_samples_split=5,random_state=1234)#第一次测试最优
classifier2.fit(X_train,y_train)
score2=classifier2.score(X_test,y_test)
print("基分类器2score:",score2)
end_time_base2=time.time()
base_time2=end_time_base2-start_time_base2
print('基分类器2(15)所用时间:',base_time2)
memory_end2 =psutil.Process(os.getpid()).memory_info().rss/1024/1024  #MB
memory_usage2=memory_end2-memory_start2
print('基分类器2(15)所用内存(MB):',memory_usage2)
#保存Model
joblib.dump(classifier2,'model_mnist5/base2_learner_mnist.pkl')
print("基分类器2pkl_size(MB):",os.path.getsize('model_mnist5/base2_learner_mnist.pkl')/1024/1024)

#Bagging2
time_bagging2,top1_bagging2,m_percent_bagging2=bagging_clf(classifier2,'model_mnist5/model2_bagging',n)
pkl_size_bagging2=get_dir_size('model_mnist5/model2_bagging')
#保存结果
import pandas as pd
c2={"基分类器个数" : n,
   "performance" : top1_bagging2,
  "时间":time_bagging2,
  "内存使用率":m_percent_bagging2,
  "pkl_size_bagging":pkl_size_bagging2}#将列表a，b转换成字典
bagging_result2=pd.DataFrame(c2)#将字典转换成为数据,框
print(bagging_result2)
bagging_result2.to_csv('model_mnist5/bagging_MNIST_result2.csv')

#Boosting2
time_boosting2,top1_boosting2,m_percent_boosting2=boosting_clf(classifier2,'model_mnist5/model2_boosting',n)
pkl_size_boosting2=get_dir_size('model_mnist5/model2_boosting')
#保存结果
import pandas as pd
cb2={"基分类器个数" : n,
   "performance" : top1_boosting2,
  "时间":time_boosting2,
  "内存使用率":m_percent_boosting2,
  "pkl_size_boosting1":pkl_size_boosting2}#将列表a，b转换成字典
boost_result2=pd.DataFrame(cb2)#将字典转换成为数据框
print(boost_result2)
boost_result2.to_csv('model_mnist5/boosting_MNIST_result2.csv')


#基分类器3
memory_start3=psutil.Process(os.getpid()).memory_info().rss/1024/1024 #MB
start_time_base3=time.time()
classifier3=DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=20,min_samples_split=5,random_state=5678)#第一次测试最优
classifier3.fit(X_train,y_train)
score3=classifier3.score(X_test,y_test)
print("基分类器3score:",score3)
end_time_base3=time.time()
base_time3=end_time_base3-start_time_base3
print('基分类器3(20)所用时间:',base_time3)
memory_end3 =psutil.Process(os.getpid()).memory_info().rss/1024/1024  #MB
memory_usage3=memory_end3-memory_start3
print('基分类器3(20)所用内存(MB):',memory_usage3)
#保存Model
joblib.dump(classifier3,'model_mnist5/base3_learner_mnist.pkl')
print("基分类器3pkl_size(MB):",os.path.getsize('model_mnist5/base3_learner_mnist.pkl')/1024/1024)

# Bagging3
time_bagging3,top1_bagging3,m_percent_bagging3=bagging_clf(classifier3,'model_mnist5/model3_bagging',n)
pkl_size_bagging3=get_dir_size('model_mnist5/model3_bagging')
#保存结果
import pandas as pd
c3={"基分类器个数" : n,
   "performance" : top1_bagging3,
  "时间":time_bagging3,
  "内存使用率":m_percent_bagging3,
  "pkl_size_bagging":pkl_size_bagging3}#将列表a，b转换成字典
bagging_result3=pd.DataFrame(c3)#将字典转换成为数据框
print(bagging_result3)
bagging_result3.to_csv('model_mnist5/bagging_MNIST_result3.csv')

# Boosting3
time_boosting3,top1_boosting3,m_percent_boosting3=boosting_clf(classifier3,'model_mnist5/model3_boosting',n)
pkl_size_boosting3=get_dir_size('model_mnist5/model3_boosting')
#保存结果
import pandas as pd
cb3={"基分类器个数" : n,
   "performance" : top1_boosting3,
  "时间":time_boosting3,
  "内存使用率":m_percent_boosting3,
  "pkl_size_boosting1":pkl_size_boosting3}#将列表a，b转换成字典
boost_result3=pd.DataFrame(cb3)#将字典转换成为数据框
print(boost_result3)
boost_result3.to_csv('model_mnist5/boosting_MNIST_result3.csv')



#基分类器4
memory_start4=psutil.Process(os.getpid()).memory_info().rss/1024/1024 #MB
start_time_base4=time.time()
classifier4=DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=10,min_samples_split=5,random_state=2023)#第一次测试最优
classifier4.fit(X_train,y_train)
score4=classifier4.score(X_test,y_test)
print("基分类器4score:",score4)
end_time_base4=time.time()
base_time4=end_time_base4-start_time_base4
print('基分类器4(10)所用时间：',base_time4)
memory_end4 =psutil.Process(os.getpid()).memory_info().rss/1024/1024  #MB
memory_usage4=memory_end4-memory_start4
print('基分类器4(10)所用内存(MB):',memory_usage4)
#保存Model
joblib.dump(classifier4,'model_mnist5/base4_learner_mnist.pkl')
print("基分类器4pkl_size(MB):",os.path.getsize('model_mnist5/base4_learner_mnist.pkl')/1024/1024)

# Bagging4
time_bagging4,top1_bagging4,m_percent_bagging4=bagging_clf(classifier4,'model_mnist5/model4_bagging',n)
pkl_size_bagging4=get_dir_size('model_mnist5/model4_bagging')
#保存结果
import pandas as pd
c4={"基分类器个数" : n,
   "performance" : top1_bagging4,
  "时间":time_bagging4,
  "内存使用率":m_percent_bagging4,
  "pkl_size_bagging":pkl_size_bagging4}#将列表a，b转换成字典
bagging_result4=pd.DataFrame(c4)#将字典转换成为数据框
print(bagging_result4)
bagging_result4.to_csv('model_mnist5/bagging_MNIST_result4.csv')

# Boosting4
time_boosting4,top1_boosting4,m_percent_boosting4=boosting_clf(classifier4,'model_mnist5/model4_boosting',n)
pkl_size_boosting4=get_dir_size('model_mnist5/model4_boosting')
#保存结果
import pandas as pd
cb4={"基分类器个数" : n,
   "performance" : top1_boosting4,
  "时间":time_boosting4,
  "内存使用率":m_percent_boosting4,
  "pkl_size_boosting1":pkl_size_boosting4}#将列表a，b转换成字典
boost_result4=pd.DataFrame(cb4)#将字典转换成为数据框
print(boost_result4)
boost_result4.to_csv('model_mnist5/boosting_MNIST_result4.csv')