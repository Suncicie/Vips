# -*- coding: utf-8 -*-
import os
os.getcwd()
os.chdir("/Users/Suncicie/Study/DataMining/Competition/Vips")
import numpy as np
import pandas as  pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from scipy import sparse
import gc
import pickle

# In[47]:

# user_action_train=pd.read_table("data/user_action_train.txt",header=None)
# goods=pd.read_table("data/goods_train.txt",header=None)
# user_action_test=pd.read_table("data/user_action_test_items.txt",header=None)

# user_action_train.columns=["user_id","spu_id","bought","date"]
# user_action_test.columns=["user_id","spu_id","prob"]
# goods.columns=["spu_id","brand_id","cat_id"]
# train=pd.merge(user_action_train,goods,on="spu_id",how="left")
# test=pd.merge(user_action_test,goods,on="spu_id",how="left")
#
# # print train.head()
# # print test.head()
#
dump_path_train="data/origin_train1.pkl"
dump_path_test="data/origin_test1.pkl"
# # pickle.dump(train, open(dump_path_train, 'w'))
# # pickle.dump(test, open(dump_path_test, 'w'))
# pd.to_pickle(train,dump_path_train)
# pd.to_pickle(test,dump_path_test)

# print train.head()
# print test.head()

# construct the feature
# split date to month day and week
# train["month"]=train["date"].map(lambda x: x[0:2])
# train["day"]=train["date"].map(lambda x:x[3:5])
# print train.head()


# In[53]:

# construct the sum of a user bought in the 3 months
# instslled_count = dfInstalled[['userID','count']].groupby('userID').sum().reset_index()
# train=pd.read_pickle(dump_path_train)
# test=pd.read_pickle(dump_path_test)
# #
# # #
# UserClickDF=train[["user_id","bought"]].groupby("user_id").count().reset_index()
# UserClickDF.columns=["user_id","click_count"]
# print UserClickDF.head()
# pickle.dump(UserClickDF, open("data/UserClickDF.pkl", 'w'))
# # #
# UserBoughtDF=train[["user_id","bought"]].groupby("user_id").sum().reset_index()
# UserBoughtDF.columns=["user_id","bought_count"]
# print UserBoughtDF.head()
# pickle.dump(UserBoughtDF, open("data/UserBoughtDF.pkl", 'w'))
# #
# #
# #
# # #
# train=pd.merge(train,UserClickDF,on="user_id",how="left")
# test=pd.merge(test,UserClickDF,on="user_id",how="left")
# # # #
# train=pd.merge(train,UserBoughtDF,on="user_id",how="left")
# test=pd.merge(test,UserBoughtDF,on="user_id",how="left")
# # #
# # #
# print train.head()
# print test.head()
# dump_path_train_temp="data/origin_train_temp.pkl"
# dump_path_test_temp="data/origin_test_temp.pkl"
# pd.to_pickle(train,dump_path_train)
# pd.to_pickle(test,dump_path_test)

# pickle.dump(train, open(dump_path_train_temp, 'w'))
# pickle.dump(test, open(dump_path_test_temp, 'w'))
# train=pd.read_pickle(dump_path_train)
# test=pd.read_pickle(dump_path_test)
# train["bought_rate"]=np.rint(((train["bought_count"]/train["click_count"])*100000000)/100000)
# test["bought_rate"]=np.rint(((test["bought_count"]/test["click_count"])*100000000)/100000)
# #
# # print train.head()
# # print test.head()
#
# # construct the sum of a user click in the 3 months
# # construct the sum of a user bought in the 3 months
# # construct the rate of bought/click in the 3 months
# # construct the mean bought of a month #this has no sense for it have sum as a feature
#
#
# # encoding="utf-8"
# # ---the feature about brand, it show whether a user like buy one brand
# # the number of a user cat brand in 3 month
# UserClickBrand=train.groupby(["user_id"])["brand_id"].value_counts()
# UserClickBrand=UserClickBrand.rename("brand_id_count").reset_index()
# print UserClickBrand.head()
# # #  user_id    brand_id  brand_id_count
# # # 0        3  10004318.0              19
# # # 1        3  10005367.0              18
# # # 2        3  10013106.0              17
# # # 3        3  10020991.0              17
# # # 4        3  10000601.0              14
# # #
# UserClickBrandCount=UserClickBrand.groupby(["user_id"])["brand_id"].count().reset_index()
# UserClickBrandCount.columns=["user_id","brand_count"]
# print UserClickBrandCount.head()
# #   # user_id  brand_id
# # # 0        3       108
# # # 1        4        60
# # # 2       11        21
# # # 3       16        17
# # # 4       17       407
# train=pd.merge(train,UserClickBrandCount,on="user_id",how="left")
# test=pd.merge(test,UserClickBrandCount,on="user_id",how="left")
# # #
# # # # ---the number of a user cat goods in 3 month
# # # # print UserClickBrand2.head()
# UserClickGoodsCount=UserClickBrand.groupby(["user_id"])["brand_id_count"].sum().reset_index()
# UserClickGoodsCount.columns=["user_id","goods_count"]
# print UserClickGoodsCount.head()
# #
# train=pd.merge(train,UserClickGoodsCount,on="user_id",how="left")
# test=pd.merge(test,UserClickGoodsCount,on="user_id",how="left")
# # print test.head()
# # print train.head()
#
#
# # ---the rate of num of goods/num of brands 买一个商品（用类别来衡量）看的牌子种类多不多
#
# train["goods/brands"]=np.rint(train["goods_count"]/train["brand_count"])
# test["goods/brands"]=np.rint(test["goods_count"]/test["brand_count"])
# #
# # print train.head()
# # print test.head()
#
# # there have an other one
#
# # ---the feature about the cat, it shows whether a user like many category
# UserClickCat=train.groupby(["user_id"])["cat_id"].value_counts()
# UserClickCat=UserClickCat.rename("cat_count").reset_index()
# print UserClickCat.head()
# #   user_id  cat_id  cat_count
# # 0        3   311.0        123
# # 1        3  1056.0         46
# # 2        3  1012.0         45
# # 3        3   297.0         36
# # 4        3   271.0         23
#
# # ---the number of a user cat id in 3 month
# # ---Don't calculate the mean sum(cat_count)/sum(cat_id)
# UserClickCatCount=UserClickCat.groupby(["user_id"])["cat_id"].count().reset_index()
# UserClickCatCount.columns=["user_id","cat_count"]
# train=pd.merge(train,UserClickCatCount,on="user_id",how="left")
# test=pd.merge(test,UserClickCatCount,on="user_id",how="left")
#
# # ---the rate of the goodsnum/catsnum
# train["goods/cats"]=np.rint(train["goods_count"]/train["cat_count"])
# test["goods/cats"]=np.rint(test["goods_count"]/test["cat_count"])
#
# # ---the feature combine the brand and the cat
# # a cat he see how many brands  # the rate of a brand ,he cat how many goods, merge on brand 这个是根据商品建模了
# train["brands/cat"]=np.rint(train["brand_count"]/train["cat_count"])
# test["brands/cat"]=np.rint(test["brand_count"]/test["cat_count"])
# print train.head()
# print test.head()
#
# pd.to_pickle(train,dump_path_train)
# pd.to_pickle(test,dump_path_test)

# there have an other one


# In[138]:

# y_train=train2["bought"]
# enc=OneHotEncoder()
# feats=["spu_id","brand_id_x","cat_id","click_count","bought_count","bought_rate","brand_id_count",
#        "goods_count","goods/brands","cat_count","goods/cats","brands/cat"]
#
# for i,feat in enumerate(feats):
#     x_train=enc.fit_transform(train2[feat].values.reshape(-1,1))
#     x_test=enc.fit_transform(test[feat].values.reshape(-1,1))
#     if i == 0:
#         X_train, X_test = x_train, x_test
#     else:
#         X_train, X_test = sparse.hstack((X_train, x_train)), sparse.hstack((X_test, x_test))
#
#
# # In[141]:
#
# x_train=enc.fit_transform(train2["user_id"].values.reshape(-1,1))
# x_test=enc.fit_transform(test["user_id"].values.reshape(-1,1))
#
# X_train, X_test = sparse.hstack((X_train, x_train)), sparse.hstack((X_test, x_test))
#
#
# # In[142]:
#
train=pd.read_pickle(dump_path_train)
test=pd.read_pickle(dump_path_test)

about_columns=["user_id","spu_id","brand_id","cat_id","click_count","bought_count","bought_rate","brand_count",
       "goods_count","goods/brands","cat_count","goods/cats","brands/cat"]


y_train=train["bought"].values
X_train=train[about_columns]
X_test=test[about_columns]


lr=LogisticRegression()
lr.fit(X_train,y_train)
proba_test=lr.predict_proba(X_test)[:,1]
print proba_test.head()
proba_test.to_csv("data/noOneHotResult.txt",seq=" ",index=False)

