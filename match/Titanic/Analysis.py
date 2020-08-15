
# coding: utf-8

# # 1. 提出问题：

# ### 什么样的人在泰坦尼克号中更容易存活?

# # 2. 理解数据

# ### 2.1 导入相关的库

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### 2.2 导入数据

# In[2]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[3]:


train.info()


# In[4]:


train.isnull().sum()


# ### 2.3 查看数据

# In[5]:


train.head()


# In[6]:


print('训练数据集：', train.shape, '测试数据集：', test.shape)


# In[7]:


full = train.append(test, ignore_index=True)
full.shape


# In[8]:


full.head()


# Cabin      : 客舱号
# Embarked   : 登船港口(1起点S:Southampton，经过点C:Cherbourg,2起点Q:Queenstown)
# Fare       : 船票价格
# Parch      : 船上父母/子女数（不同代直系亲属数）
# PassengerId: 乘客编号 
# Pclass     : 客舱等级（1表示1等舱）
# SibSp      : 船上兄弟姐妹数/配偶数（同代直系亲属数）
# Survived   : 生存情况（1=存活， 0=死亡）
# Ticket     : 船票编号

# In[9]:


full.describe()


# # 3. 数据清洗

# In[10]:


full.info()


# In[11]:


full.dtypes


# ### 3.1 数据预处理：缺失数据处理

# In[12]:


# 查看数据各列都有多少缺失值
full.isnull().sum().sort_values()


# In[13]:


# 一般采用现有数据的平均值来填充缺失数据
full['Age'] = full.Age.fillna(full.Age.mean())
full['Fare'] = full.Fare.fillna(full.Fare.mean())
full.isnull().sum().sort_values()


# In[14]:


# 导入计数器
from collections import Counter
print(Counter(full.Embarked)) # 统计Embarked列中各值的个数


# In[15]:


# 填充缺损值
full['Embarked'] = full.Embarked.fillna('S')   # 用'S'来填充Embarked列中的缺损值(NaN)
full['Cabin'] = full.Cabin.fillna('U')


# In[16]:


full.isnull().sum().sort_values()


# ### 3.2 特征提取

# In[17]:


# 通过map将性别映射为数值1与0
sex_mapDict = {'male':1, 'female':0}
full['Sex'] = full.Sex.map(sex_mapDict)
full.head()


# In[18]:


#使用get_dummies进行one-hot编码，产生虚拟变量（dummy variables）,列名前缀是Embarked
embarkedDf = pd.DataFrame()
embarkedDf = pd.get_dummies(full.Embarked, prefix='Embarked')
embarkedDf.head()


# In[19]:


# 添加one-hot编码产生的虚拟变量到数据集full
full = pd.concat([full, embarkedDf],axis=1)
# 删除源登录港口
full.drop('Embarked', axis=1, inplace=True)
full.head()


# In[20]:


PclassDf = pd.DataFrame()
PclassDf = pd.get_dummies(full.Pclass, prefix='Pclass')
full = pd.concat([full, PclassDf], axis=1)
full.drop('Pclass', axis=1, inplace=True)
full.head()


# In[21]:


# 定义函数：从姓名中获取头衔
def getTitle(name):
    str1 = name.split(',')[1] # 得到 Mr. Owen Harris
    str2 = str1.split('.')[0] # 得到 Mr
    str3 = str2.strip() #去除字符串头尾无效字符：空格等
    return str3


# In[22]:


# 存放提取后的特征
titleDf = pd.DataFrame()
titleDf['Title'] = full['Name'].map(getTitle)
titleDf.head()


# In[23]:


titleDf.Title.value_counts()

定义以下几种头衔类别：
Officer : 政府官员
Royalty : 王室
Mr      : 已婚男士
Mrs     : 已婚女士
Miss    : 年轻未婚女子
Master  ：有技能的人/教师
# In[25]:


# 姓名中头衔字符串与定义头衔类别的映射关系
title_mapDict = {
    'Capt':'Officer',
    'Col':'Officer',
    'Major':'Officer',
    'Jonkheer':'Royalty',
    'Don':'Royalty',
    'Sir':'Royalty',
    'Dr':'Officer',
    'Rev':'Officer',
    'the Countess':'Royalty',
    'Dona':'Royalty',
    'Mme':'Mrs',
    'Mlle':'Miss',
    'Ms':'Mrs',
    'Mr':'Mr',
    'Mrs':'Mrs',
    'Miss':'Miss',
    'Master':'Master',
    'Lady':'Royalty'
}


# In[26]:


# map函数对Series每个数据应用自定义函数
titleDf['Title'] = titleDf.Title.map(title_mapDict)
titleDf = pd.get_dummies(titleDf.Title)
titleDf.head()


# In[27]:


# 使用concat方法拼接，按列拼接（axis=1）
full = pd.concat([full, titleDf], axis=1)
full.drop('Name', axis=1, inplace=True)
full.head()


# In[28]:


cabinDf = pd.DataFrame()
full['Cabin'] = full.Cabin.map(lambda c:c[0])
cabinDf = pd.get_dummies(full.Cabin, prefix='Cabin')
cabinDf.head()


# In[29]:


full = pd.concat([full, cabinDf], axis=1)
full.drop('Cabin', axis=1, inplace=True)
full.head()


# In[30]:


# 存放家庭信息
familyDf = pd.DataFrame()
familyDf['FamilySize'] = full.Parch + full.SibSp + 1 # 最后的 +1 是加上自己
'''
家庭类别：
小家庭 Family_Single:家庭人数=1
中等家庭Family_Small:2 <= 家庭人数 <= 4
大家庭Family——Large:家庭人数 > 4
'''
familyDf['Family_Single'] = familyDf.FamilySize.map(lambda s : 1 if s == 1 else 0)
familyDf['Family_Small'] = familyDf.FamilySize.map(lambda s : 1 if 2 <= s <= 4 else 0)
familyDf['Family_Large'] = familyDf.FamilySize.map(lambda s : 1 if s >4 else 0)
familyDf.head()


# In[31]:


full = pd.concat([full,familyDf],axis=1)
full.head()


# In[32]:


full.shape


# ### 3.3 特征选择

# In[33]:


# 通过corr方法求出df的各列两两之间的相关性矩阵
corrDf = full.corr()
# 查看各特征与生存情况的相关系数，按降序排列
corrDf['Survived'].sort_values(ascending=False)


# In[34]:


corrDf['Survived'].sort_values(ascending=False).plot(kind = 'bar')


# In[35]:


# 特征选择
full_X = pd.concat([
    titleDf,#头衔
    PclassDf,#客舱等级
    familyDf,#家庭大小
    full['Fare'],#船票价格
    cabinDf,#船舱号
    embarkedDf,#登船港口
    full['Sex'],#性别
],axis=1)
full_X.head()


# # 4.构建模型和评估模型

# In[36]:


source_XData = full_X.loc[:890, :] # 选取训练数据集部分
pre_XData = full_X.loc[891:, :] # 选取测试数据集部分
source_YData = full.loc[:890, 'Survived'] # 选取训练数据集中的'Survived'列


# In[37]:


print('训练集X大小=', source_XData.shape, '；训练集y大小=', source_YData.shape, '；测试集X大小=', pre_XData.shape)


# In[38]:


full_X.info()


# # 逻辑回归

# In[39]:


#from sklearn.cross_validation import train_test_split # 已弃用
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[40]:


list_linear = []
for i in range(1000):
    train_x, test_x, train_y, test_y = train_test_split(source_XData, source_YData, train_size=0.8)
    reg_1 = LogisticRegression()
    reg_1.fit(train_x, train_y)
    list_linear.append(reg_1.score(test_x, test_y))
S_linear = pd.Series(list_linear)
S_linear.describe()


# # 随机森林

# In[41]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# In[42]:


list_Random = []
for i in range(1000):
    train_x, test_x, train_y, test_y = train_test_split(source_XData, source_YData, train_size=0.8)
    reg_2 = RandomForestClassifier(n_estimators=100)
    reg_2.fit(train_x, train_y)
    list_Random.append(reg_2.score(test_x, test_y))
S_Random = pd.Series(list_Random)
S_Random.describe()


# # 支持向量机

# In[43]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC


# In[44]:


list_SVC = []
for i in range(100):
    train_x, test_x, train_y, test_y = train_test_split(source_XData, source_YData, train_size=0.8)
    reg_3 = SVC()
    reg_3.fit(train_x, train_y)
    list_SVC.append(reg_3.score(test_x, test_y))
S_SVC = pd.Series(list_Random)
S_SVC.describe()


# # Gradient Boosting Classifier

# In[45]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier


# In[46]:


list_Gradient = []
for i in range(100):
    train_x, test_x, train_y, test_y = train_test_split(source_XData, source_YData, train_size=0.8)
    reg_4 = GradientBoostingClassifier()
    reg_4.fit(train_x, train_y)
    list_Gradient.append(reg_4.score(test_x, test_y))
S_Gradient = pd.Series(list_Gradient)
S_Gradient.describe()


# # K-nearest neighbors

# In[47]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# In[48]:


list_K = []
for i in range(100):
    train_x, test_x, train_y, test_y = train_test_split(source_XData, source_YData, train_size=0.8)
    reg_5 = KNeighborsClassifier(n_neighbors = 3)
    reg_5.fit(train_x, train_y)
    list_K.append(reg_5.score(test_x, test_y))
S_K = pd.Series(list_K)
S_K.describe()


# # 贝叶斯

# In[49]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


# In[50]:


list_bays = []
for i in range(100):
    train_x, test_x, train_y, test_y = train_test_split(source_XData, source_YData, train_size=0.8)
    reg_6 = GaussianNB()
    reg_6.fit(train_x, train_y)
    list_bays.append(reg_6.score(test_x, test_y))
S_bays = pd.Series(list_bays)
S_bays.describe()


# # 5.方案撰写

# In[51]:


preDf = pd.DataFrame()
preDf['PassengerId'] = full.loc[891:,'PassengerId']


# In[52]:


pre_XData.info()


# In[53]:


preDf['Survived'] = reg_1.predict(pre_XData)
preDf['Survived'] = preDf['Survived'].astype(int)
preDf.head()


# In[54]:


preDf.to_csv('My_submission.csv', index = False)

