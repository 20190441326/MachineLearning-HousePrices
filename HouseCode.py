#!/usr/bin/env python
# coding: utf-8

# ## 导包

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling as ppf #探索性数据分析（EDA）
import warnings##忽略警告
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')


# In[2]:


from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import LabelEncoder#标签编码
from sklearn.preprocessing import RobustScaler, StandardScaler#去除异常值与数据标准化
from sklearn.pipeline import Pipeline, make_pipeline#构建管道
from scipy.stats import skew#偏度
from sklearn.impute import SimpleImputer


# ## 读取并查看原数据

# In[3]:


train = pd.read_csv(r"G:\study类\大三上\机器学习\课程设计\datas\train.csv") #将数据读取进来


# In[4]:


test = pd.read_csv(r"G:\study类\大三上\机器学习\课程设计\datas\test.csv") #将数据读取进来


# In[5]:


train.head()#默认显示前五行


# In[6]:


test.head()

从中可以看出还是有很多数据需要处理的
# ## 数据探索性分析 pandas_profiling

# In[7]:


ppf.ProfileReport(train)


# In[8]:


train.YearBuilt#显示这一列的数据


# In[9]:


train.SalePrice


# ## 通过箱型图查看异常值，离群点

# In[10]:


plt.figure(figsize=(12,8))
sns.boxplot(train.YearBuilt, train.SalePrice)


# ## 通过散点图来观察存在线型的关系

# In[11]:


plt.figure(figsize=(12,6))
plt.scatter(x=train.GrLivArea, y=train.SalePrice)
plt.xlabel("GrLivArea", fontsize=13)
plt.ylabel("SalePrice", fontsize=13)
plt.ylim(0,800000)


# ## 把太偏离线性的那些数据给去掉，把对应的索引给删掉

# In[12]:


train.drop(train[(train["GrLivArea"]>4000)&(train["SalePrice"]<300000)].index,inplace=True)

删除后的图像
# In[13]:


plt.figure(figsize=(12,6))
plt.scatter(x=train.GrLivArea, y=train.SalePrice)
plt.xlabel("GrLivArea", fontsize=13)
plt.ylabel("SalePrice", fontsize=13)
plt.ylim(0,800000)


# ### 把test数据也做相同的处理

# In[14]:


full = pd.concat([train,test],ignore_index=True)


# ### 因为ID列和索引值都相同，故这里把ID列给删掉

# In[15]:


full.drop("Id",axis=1,inplace=True)


# In[16]:


full.head()#查看删除列之后的值


# In[17]:


full.info()#查看删除后的数据信息


# # 数据清洗--空值的填充、删除

# #### 查看缺失值，并且缺失的个数要从高到低排序

# In[18]:


miss = full.isnull().sum()#统计出空值的个数


# In[19]:


miss


# In[20]:


miss[miss>0].sort_values(ascending=True)#由低到高进行排序


# In[21]:


full.info() #查看数据信息


# ## 空值的填充与删除

# 对字符类型的进行填充

# In[22]:


cols1 = ["PoolQC" , "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageQual", "GarageCond", "GarageFinish", "GarageYrBlt", "GarageType", "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType2", "BsmtFinType1", "MasVnrType"]
for col in cols1:
    full[col].fillna("None",inplace=True)


# In[23]:


full.head()


# 对数值类型的进行填充

# In[24]:


cols=["MasVnrArea", "BsmtUnfSF", "TotalBsmtSF", "GarageCars", "BsmtFinSF2", "BsmtFinSF1", "GarageArea"]
for col in cols:
    full[col].fillna(0, inplace=True)


# 对lotfrontage的空值使用其均值进行填充

# In[25]:


full["LotFrontage"].fillna(np.mean(full["LotFrontage"]),inplace=True)


# 对下面的列使用众数进行填充

# In[26]:


cols2 = ["MSZoning", "BsmtFullBath", "BsmtHalfBath", "Utilities", "Functional", "Electrical", "KitchenQual", "SaleType","Exterior1st", "Exterior2nd"]
for col in cols2:
    full[col].fillna(full[col].mode()[0], inplace=True)


# 查看是否还有未填充好的数据

# In[27]:


full.isnull().sum()[full.isnull().sum()>0]

发现只有test的没有标签列，故已经把数据中的空值处理好了
# ## 数据预处理--把字符变成数值型

# In[28]:


full["MSZoning"].mode()[0]


# In[29]:


pd.set_option('display.max_rows', None)  # 设置显示最大行，不然有一些数据会以“...”显示，不能看到部分数据
full.MSZoning

从上面可以发现有一些数据，比如31行：C（all），需要把这些数据转换成字符串的形式，将一些数字特征转换为类别特征,使用LabelEncoder来实现
# In[30]:


for col in cols2:
    full[col]=full[col].astype(str)##astype来进行数据转换成字符串类型


# In[31]:


lab = LabelEncoder() #对不连续的数字或者文本进行编号


# #### 把下列内容字符型转换为数字型

# In[32]:


full["Alley"] = lab.fit_transform(full.Alley)
full["PoolQC"] = lab.fit_transform(full.PoolQC)
full["MiscFeature"] = lab.fit_transform(full.MiscFeature)
full["Fence"] = lab.fit_transform(full.Fence)
full["FireplaceQu"] = lab.fit_transform(full.FireplaceQu)
full["GarageQual"] = lab.fit_transform(full.GarageQual)
full["GarageCond"] = lab.fit_transform(full.GarageCond)
full["GarageFinish"] = lab.fit_transform(full.GarageFinish)
full["GarageYrBlt"] = full["GarageYrBlt"].astype(str)
full["GarageYrBlt"] = lab.fit_transform(full.GarageYrBlt)
full["GarageType"] = lab.fit_transform(full.GarageType)
full["BsmtExposure"] = lab.fit_transform(full.BsmtExposure)
full["BsmtCond"] = lab.fit_transform(full.BsmtCond)
full["BsmtQual"] = lab.fit_transform(full.BsmtQual)
full["BsmtFinType2"] = lab.fit_transform(full.BsmtFinType2)
full["BsmtFinType1"] = lab.fit_transform(full.BsmtFinType1)
full["MasVnrType"] = lab.fit_transform(full.MasVnrType)
full["BsmtFinType1"] = lab.fit_transform(full.BsmtFinType1)


# In[33]:


full.head()


# 将一些未转换的列继续转换为数字型

# In[34]:


full["MSZoning"] = lab.fit_transform(full.MSZoning)
full["BsmtFullBath"] = lab.fit_transform(full.BsmtFullBath)
full["BsmtHalfBath"] = lab.fit_transform(full.BsmtHalfBath)
full["Utilities"] = lab.fit_transform(full.Utilities)
full["Functional"] = lab.fit_transform(full.Functional)
full["Electrical"] = lab.fit_transform(full.Electrical)
full["KitchenQual"] = lab.fit_transform(full.KitchenQual)
full["SaleType"] = lab.fit_transform(full.SaleType)
full["Exterior1st"] = lab.fit_transform(full.Exterior1st)
full["Exterior2nd"] = lab.fit_transform(full.Exterior2nd)


# In[35]:


full.head()


# #### 发现还有一些列是字符型，未能完全转换为数字型

# In[36]:


full.drop("SalePrice",axis=1,inplace=True)##删除这一列，以便后面进行操作


# #### 从结果可以看出，行和列变得很多了

# #### 可以看到所有数据都显示为数字型了

# In[37]:


##自己写一个转换函数
class labelenc(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self,X,y=None):
        return self
    ##对三个年份来进行一个标签编码,这里可以随便自己添加，这里可以随便加的
    def transform(self,X):
        lab=LabelEncoder()
        X["YearBuilt"] = lab.fit_transform(X["YearBuilt"])
        X["YearRemodAdd"] = lab.fit_transform(X["YearRemodAdd"])
        X["GarageYrBlt"] = lab.fit_transform(X["GarageYrBlt"])
        X["BldgType"] = lab.fit_transform(X["BldgType"])
        
        return X


# In[38]:


#写一个转换函数
class skew_dummies(BaseEstimator, TransformerMixin):
    def __init__(self,skew=0.5):#偏度
        self.skew = skew
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X_numeric=X.select_dtypes(exclude=["object"])#而是去除了包含了对象数据类型，取出来绝大部分是数值型，取出字符类型的数据
        skewness = X_numeric.apply(lambda x: skew(x))#匿名函数，做成字典的形式
        skewness_features = skewness[abs(skewness) >= self.skew].index#通过条件来涮选出skew>=0.5的索引的条件，取到了全部数据，防止数据的丢失
        X[skewness_features] = np.log1p(X[skewness_features])#求对数，进一步让他更符合正态分布
        X = pd.get_dummies(X)##一键独热，独热编码，（试错经历），也可以不要
        return X


# In[39]:


# 构建管道
pipe = Pipeline([#构建管道
    ('labenc', labelenc()),
    ('skew_dummies', skew_dummies(skew=2)),
    ])


# In[40]:


# 保存原来的数据以备后用，为了防止写错
full2 = full.copy()


# In[41]:


pipeline_data = pipe.fit_transform(full2)


# In[42]:


pipeline_data.shape


# In[43]:


pipeline_data.head()


# In[44]:


n_train=train.shape[0]#训练集的行数
X = pipeline_data[:n_train]#取出处理之后的训练集
test_X = pipeline_data[n_train:]#取出n_train后的数据作为测试集
y= train.SalePrice
X_scaled = StandardScaler().fit(X).transform(X)#做转换
y_log = np.log(train.SalePrice)#使其更符合正态分布
#得到测试集
test_X_scaled = StandardScaler().fit_transform(test_X)


# In[45]:


from sklearn.linear_model import Lasso#运用算法来进行训练集以得到特征的重要性
lasso=Lasso(alpha=0.001)
lasso.fit(X_scaled,y_log)


# In[46]:


FI_lasso = pd.DataFrame({"Feature Importance":lasso.coef_}, index=pipeline_data.columns)#索引和重要性做成dataframe形式


# In[47]:


FI_lasso.sort_values("Feature Importance",ascending=False)#由高到低进行排序


# In[48]:


#可视化
FI_lasso[FI_lasso["Feature Importance"]!=0].sort_values("Feature Importance").plot(kind="barh",figsize=(15,25))#barh：把x，y轴反转
plt.xticks(rotation=90)
plt.show()#画图显示


# ##  得到特征重要性图之后就可以进行特征选择与重做

# In[49]:


class add_feature(BaseEstimator, TransformerMixin):#定义转换函数
    def __init__(self,additional=1):
        self.additional = additional
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        if self.additional==1:
            X["TotalHouse"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]   
            X["TotalArea"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"]
            
        else:
            X["TotalHouse"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]   
            X["TotalArea"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"]
            
            X["+_TotalHouse_OverallQual"] = X["TotalHouse"] * X["OverallQual"]
            X["+_GrLivArea_OverallQual"] = X["GrLivArea"] * X["OverallQual"]
            X["+_oMSZoning_TotalHouse"] = X["oMSZoning"] * X["TotalHouse"]
            X["+_oMSZoning_OverallQual"] = X["oMSZoning"] + X["OverallQual"]
            X["+_oMSZoning_YearBuilt"] = X["oMSZoning"] + X["YearBuilt"]
            X["+_oNeighborhood_TotalHouse"] = X["oNeighborhood"] * X["TotalHouse"]
            X["+_oNeighborhood_OverallQual"] = X["oNeighborhood"] + X["OverallQual"]
            X["+_oNeighborhood_YearBuilt"] = X["oNeighborhood"] + X["YearBuilt"]
            X["+_BsmtFinSF1_OverallQual"] = X["BsmtFinSF1"] * X["OverallQual"]
            
            X["-_oFunctional_TotalHouse"] = X["oFunctional"] * X["TotalHouse"]
            X["-_oFunctional_OverallQual"] = X["oFunctional"] + X["OverallQual"]
            X["-_LotArea_OverallQual"] = X["LotArea"] * X["OverallQual"]
            X["-_TotalHouse_LotArea"] = X["TotalHouse"] + X["LotArea"]
            X["-_oCondition1_TotalHouse"] = X["oCondition1"] * X["TotalHouse"]
            X["-_oCondition1_OverallQual"] = X["oCondition1"] + X["OverallQual"]
            
           
            X["Bsmt"] = X["BsmtFinSF1"] + X["BsmtFinSF2"] + X["BsmtUnfSF"]
            X["Rooms"] = X["FullBath"]+X["TotRmsAbvGrd"]
            X["PorchArea"] = X["OpenPorchSF"]+X["EnclosedPorch"]+X["3SsnPorch"]+X["ScreenPorch"]
            X["TotalPlace"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"] + X["OpenPorchSF"]+X["EnclosedPorch"]+X["3SsnPorch"]+X["ScreenPorch"]

    
            return X


# In[50]:


pipe = Pipeline([#把后面的东西加到管道里面来
    ('labenc', labelenc()),
    ('add_feature', add_feature(additional=2)),
    ('skew_dummies', skew_dummies(skew=4)),
    ])


# In[51]:


pipe


# In[52]:


n_train=train.shape[0]#训练集的行数
X = pipeline_data[:n_train]#取出处理之后的训练集
test_X = pipeline_data[n_train:]#取出n_train后的数据作为测试集
y= train.SalePrice
X_scaled = StandardScaler().fit(X).transform(X)#做转换
y_log = np.log(train.SalePrice)##这里要注意的是，更符合正态分布
#得到测试集
test_X_scaled = StandardScaler().fit_transform(test_X)


# ## 模型的构建

# #### 线性回归

# In[53]:


from sklearn.tree import DecisionTreeRegressor#导入模型


# In[54]:


#model = LinearRegression()#导入模型


# In[55]:


#model1 =model.fit(X_scaled,y)#开始训练


# In[56]:


model = DecisionTreeRegressor()


# In[57]:


model1 =model.fit(X_scaled,y_log)


# ## 前期比较简单的处理得到结果，并没有进行模型的堆叠

# In[58]:


#predict = modexp.predict(test_x)


# In[59]:


# result=pd.DataFrame({'Id':test.Id, 'SalePrice':predict})
# result.to_csv("submission1.csv",index=False)


# In[60]:


# predict = np.exp(model1.predict(test_X_scaled))#np.exp是对上面的对数变换之后的反变换


# In[61]:


# result=pd.DataFrame({'Id':test.Id, 'SalePrice':predict})
# result.to_csv("submission.csv",index=False)


# ## 模型的堆叠与集成并且选择最优参数，模型和评估方式

# In[62]:


from sklearn.model_selection import cross_val_score, GridSearchCV, KFold#交叉验证，网格搜索，k折验证
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor


# In[63]:


#定义交叉验证的策略，以及评估函数
def rmse_cv(model,X,y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))#交叉验证
    return rmse


# In[64]:


models = [LinearRegression(),Ridge(),Lasso(alpha=0.01,max_iter=10000),RandomForestRegressor(),GradientBoostingRegressor(),SVR(),LinearSVR(),
          ElasticNet(alpha=0.001,max_iter=10000),SGDRegressor(max_iter=1000,tol=1e-3),BayesianRidge(),KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),
          ExtraTreesRegressor(),XGBRegressor()]#这里也是列表


# In[65]:


names = ["LR", "Ridge", "Lasso", "RF", "GBR", "SVR", "LinSVR", "Ela","SGD","Bay","Ker","Extra","Xgb"]#列表
for name, model in zip(names, models):
    score = rmse_cv(model, X_scaled, y_log)
    print("{}: {:.6f}, {:.4f}".format(name,score.mean(),score.std()))


# In[66]:


#定义交叉方式，先指定模型后指定参数，方便测试多个模型，网格交叉验证
class grid():
    def __init__(self,model):
        self.model = model#导入模型
    #所有模型进行验证5次
    def grid_get(self,X,y,param_grid):#网格参数一般做出字典的格式
        grid_search = GridSearchCV(self.model,param_grid,cv=5, scoring="neg_mean_squared_error")
        grid_search.fit(X,y)
        print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))
        grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])
        print(pd.DataFrame(grid_search.cv_results_)[['params','mean_test_score','std_test_score']])


# In[67]:


grid(Lasso()).grid_get(X_scaled,y_log,{'alpha': [0.0004,0.0005,0.0007,0.0006,0.0009,0.0008],'max_iter':[10000]})


# In[68]:


grid(Ridge()).grid_get(X_scaled,y_log,{'alpha':[35,40,45,50,55,60,65,70,80,90]})


# In[69]:


grid(SVR()).grid_get(X_scaled,y_log,{'C':[11,12,13,14,15],'kernel':["rbf"],"gamma":[0.0003,0.0004],"epsilon":[0.008,0.009]})#支持向量机回归


# In[70]:


param_grid={'alpha':[0.2,0.3,0.4,0.5], 'kernel':["polynomial"], 'degree':[3],'coef0':[0.8,1,1.2]}#定义好的参数，用字典来表示
grid(KernelRidge()).grid_get(X_scaled,y_log,param_grid)


# In[71]:


grid(ElasticNet()).grid_get(X_scaled,y_log,{'alpha':[0.0005,0.0008,0.004,0.005],'l1_ratio':[0.08,0.1,0.3,0.5,0.7],'max_iter':[10000]})


# In[72]:


#定义加权平均值，就相当于自己写fit_transform（）
class AverageWeight(BaseEstimator, RegressorMixin):
    def __init__(self,mod,weight):
        self.mod = mod#模型的个数
        self.weight = weight#权重
        
    def fit(self,X,y):
        self.models_ = [clone(x) for x in self.mod]
        for model in self.models_:
            model.fit(X,y)
        return self
    
    def predict(self,X):
        w = list()
        pred = np.array([model.predict(X) for model in self.models_])
        # 针对于每一个数据点，单一的模型是乘以权重，然后加起来
        for data in range(pred.shape[1]):#取列数
            single = [pred[model,data]*weight for model,weight in zip(range(pred.shape[0]),self.weight)]
            w.append(np.sum(single))
        return w


# In[73]:


#指定每一个算法的参数
lasso = Lasso(alpha=0.0005,max_iter=10000)
ridge = Ridge(alpha=60)
svr = SVR(gamma= 0.0004,kernel='rbf',C=13,epsilon=0.009)
ker = KernelRidge(alpha=0.2 ,kernel='polynomial',degree=3 , coef0=0.8)
ela = ElasticNet(alpha=0.005,l1_ratio=0.08,max_iter=10000)
bay = BayesianRidge()


# In[74]:


#6个权重
w1 = 0.02
w2 = 0.2
w3 = 0.25
w4 = 0.3
w5 = 0.03
w6 = 0.2


# In[75]:


weight_avg = AverageWeight(mod = [lasso,ridge,svr,ker,ela,bay],weight=[w1,w2,w3,w4,w5,w6])


# In[76]:


rmse_cv(weight_avg,X_scaled,y_log),  rmse_cv(weight_avg,X_scaled,y_log).mean()#计算出交叉验证的均值


# ## 模型的堆叠

# In[77]:


class stacking(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self,mod,meta_model):
        self.mod = mod
        self.meta_model = meta_model#元模型
        self.kf = KFold(n_splits=5, random_state=42, shuffle=True)#5折的划分
        #数据集平均分成5份
    def fit(self,X,y):
        self.saved_model = [list() for i in self.mod]#用模型来进行拟合
        oof_train = np.zeros((X.shape[0], len(self.mod)))
        
        for i,model in enumerate(self.mod):#返回的是索引和模型本身
            for train_index, val_index in self.kf.split(X,y):##返回的是数据本省
                renew_model = clone(model)#模型的复制
                renew_model.fit(X[train_index], y[train_index])#对数据进行训练
                self.saved_model[i].append(renew_model)#把模型添加进去
                oof_train[val_index,i] = renew_model.predict(X[val_index])#用来预测验证集
        
        self.meta_model.fit(oof_train,y)#元模型
        return self
    
    def predict(self,X):
        whole_test = np.column_stack([np.column_stack(model.predict(X) for model in single_model).mean(axis=1) 
                                      for single_model in self.saved_model]) #得到的是整个测试集
        return self.meta_model.predict(whole_test)#返回的是利用元模型来对整个测试集进行预测
    #预测，使整个测试集
    def get_oof(self,X,y,test_X):
        oof = np.zeros((X.shape[0],len(self.mod)))#初始化为0
        test_single = np.zeros((test_X.shape[0],5))#初始化为0 
        test_mean = np.zeros((test_X.shape[0],len(self.mod)))
        for i,model in enumerate(self.mod):#i是模型
            for j, (train_index,val_index) in enumerate(self.kf.split(X,y)):#j是所有划分好的的数据
                clone_model = clone(model)#克隆模块，把模型复制一下
                clone_model.fit(X[train_index],y[train_index])#把分割好的数据进行训练
                oof[val_index,i] = clone_model.predict(X[val_index])#对验证集进行预测
                test_single[:,j] = clone_model.predict(test_X)#对测试集进行预测
            test_mean[:,i] = test_single.mean(axis=1)#测试集算好均值
        return oof, test_mean


# In[78]:


#经过预处理之后才能放到堆叠的模型里面去计算

a = SimpleImputer().fit_transform(X_scaled)#x
b = SimpleImputer().fit_transform(y_log.values.reshape(-1,1)).ravel()#y
# a = Imputer().fit_transform(X_scaled)#相当于x
# b = Imputer().fit_transform(y_log.values.reshape(-1,1)).ravel()#相当于y


# In[79]:


stack_model = stacking(mod=[lasso,ridge,svr,ker,ela,bay],meta_model=ker)#定义了第一层的和第二层的模型


# In[80]:


print(rmse_cv(stack_model,a,b))#运用了评估函数
print(rmse_cv(stack_model,a,b).mean())


# In[81]:


X_train_stack, X_test_stack = stack_model.get_oof(a,b,test_X_scaled)#将数据进行变换


# In[82]:


X_train_stack.shape, a.shape


# In[83]:


X_train_add = np.hstack((a,X_train_stack))
X_test_add = np.hstack((test_X_scaled,X_test_stack))
X_train_add.shape, X_test_add.shape


# In[84]:


print(rmse_cv(stack_model,X_train_add,b))
print(rmse_cv(stack_model,X_train_add,b).mean())


# In[85]:


stack_model = stacking(mod=[lasso,ridge,svr,ker,ela,bay],meta_model=ker)


# In[86]:


stack_model.fit(a,b)#模型进行训练


# In[87]:


pred = np.exp(stack_model.predict(test_X_scaled))#进行预测


# In[88]:


result=pd.DataFrame({'Id':test.Id, 'SalePrice':pred})
result.to_csv("submission3.csv",index=False)

