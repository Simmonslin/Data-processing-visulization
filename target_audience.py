import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
import chart_studio as cs
import plotly.graph_objs as go
from plotly.offline import iplot , init_notebook_mode
import cufflinks


product_train=pd.read_csv("order_products__train.csv")
aisle=pd.read_csv("aisles.csv")
departments=pd.read_csv("departments.csv")
products=pd.read_csv("products.csv")
orders=pd.read_csv("orders.csv")
product_prior=pd.read_csv("order_products__prior.csv")

# 藉由%matplotlib inline可直接省略plt.show()

#%%
%matplotlib inline

# value_counts()計算DataFrame中某欄位的不同值數量

# cnt_sns為Series, 可藉由.index 與.values 讀取 

cnt_sts=orders.loc[:,"eval_set"].value_counts()

# 選擇顏色
color=sns.color_palette()

#%%
plt.figure(figsize=(12,8))
sns.barplot(x=cnt_sts.index,y=cnt_sts.values,alpha=0.9,color=color[0])
plt.ylabel('Number of occurence' , fontsize = 12)
plt.xlabel('Eval set type' , fontsize = 12)
plt.title('Count of row in each dataset' , fontsize = 15)

#%%

# 虛變數用法
def get_unique_count(y):
    return len(np.unique(y))

cnt_srs = orders.groupby('eval_set')['user_id'].aggregate(get_unique_count)
total_customer=cnt_srs[0]+cnt_srs[1]+cnt_srs[2]
#%%

#Dataframe.aggregate()函數用於在一個或多個列上應用聚合

# 計算總顧客數量 
"""

原理:
    
用user_id跟eval_set分群計算，計算出各分類顧客數量

user_id 代表顧客訂單編號，因此會出現很多列相同的 user_id，因此必須先用unique

從多個相同數字變為剩1個  -->e.g: 50個 user_id 5101 unique()後剩1個 user_id 5101

"""


def get_user_dow(y):
    return len(np.unique(y))

eval_set_hour=orders.groupby("eval_set")["order_id"].aggregate(get_user_dow)

#%%

# 計算不同客戶下單平均購買量
avg_order_numbers=orders.groupby("user_id")["order_number"].aggregate(np.mean)

plt.figure(figsize=(12,8))

sns.barplot(x=avg_order_numbers.index[0:20],y=avg_order_numbers.values[0:20],color=color[3])

plt.xlabel("USER ID",fontsize=12)
plt.ylabel("Average User Order Numbers",fontsize=12)
plt.title("不同客戶下單平均購買量")

#%%

# 取得欄位名稱
print(orders.columns)

#%% 計算週一 ~ 週日顧客人數
plt.figure(figsize=(12,8))

# 運用vector 概念(類似R)
# countplot 計數圖用法 : .countplot(x=劃分單位,y=不同類別,data=數據來源的數據集)
sns.countplot(x="order_dow", data=orders,color=color[0])
plt.xlabel("Day of Week",fontsize=12)
plt.ylabel("Count on customers",fontsize=12)
plt.title("Distribution for customers",fontsize=16)

#%%  計算各時段顧客人數

plt.figure(figsize=(24,8))
sns.countplot(x="order_hour_of_day",data=orders,color=color[2])
plt.xlabel("hour of day",fontsize=13)
plt.ylabel("Count on customers",fontsize=13)
plt.title("customers over every hour",fontsize=15)

#%% 使用熱點圖檢視  (用於查看變數之間的關係)
customer_group=orders.groupby(["order_dow","order_hour_of_day"])["order_number"].aggregate("count").reset_index()

#pivot (樞紐)使用 

#DataFrame.pivot(index=None, columns=None, values=None)

# 熱點圖中縱軸為pivot
customer_group=customer_group.pivot(index="order_dow",columns="order_hour_of_day",values="order_number")

plt.figure(figsize=(24,8))
sns.heatmap(customer_group)
plt.title("Customer over every hour",fontsize=15)

#  結論: 禮拜六的下午、禮拜天的早晨是最多人的時間

#%%  幾天後會再次出門買東西

plt.figure(figsize=(15,8))
sns.countplot(x="days_since_prior_order",data=orders,color=color[3])
plt.title("Frequency of people buying food",fontsize=15)
plt.xlabel("days",fontsize=13)
plt.ylabel("count 0n customers",fontsize=13)

# 偵測異常資料

"""
.mode() Get the mode(s) of each element along the selected axis.

The mode of a set of values is the value that appears most often. It can be multiple values.

Series.value_counts -->
Return the counts of values in a Series.

"""

def detect_potential_exception(df , col):
    """
    Input:
        df:DataFrame
        col: Column name , it must be the continuous variable!
    Output:
        Detect result
    """
    confident_value = abs( ( df[col].mode().iloc[0,] - df[col].median())  / (df[col].quantile(0.75) - df[col].quantile(0.25) ))
    confident_value = round(confident_value , 2)
    if confident_value > 0.8:
        print('According to experience rule , Its is dangerous!' , confident_value)
    else:
        print('Safe!' , confident_value)
detect_potential_exception(orders, 'days_since_prior_order')


#%%
def make_skew_transform(df , feature):
    """
    To transform high skew data 
    Input:
        df:DataFrame
        feature:The columns of Variable to predict Y
    Output:
        X_rep : DataFrame which process the data with log transform
    
    """
    skew_var_x = {}
    X_rep = df.copy()
    var_x = df[feature]
    
    # 計算偏度 .skew()    
    for feature in var_x:
        skew_var_x[feature] = abs(X_rep[feature].skew())
        
    skew = pd.Series(skew_var_x).sort_values(ascending = False)
    print(skew)
    
    var_x_ln = skew.index[skew > 1]
    print('針對偏度大於1的進行對數運算')
    
    for var in var_x_ln:
        #針對小於0的我們先確保讓他大於0，平移資料
        if min(X_rep[var]) <= 0:
            X_rep[var] = np.log(X_rep[var] + abs(min(X_rep[var] + 0.01)))
        else:
            X_rep[var] = np.log(X_rep[var])
    return X_rep

#%%  計算復購率

# product_prior.shape[0]== rows長度

print(product_prior["reordered"].sum() / product_prior.shape[0])
print(product_train["reordered"].sum() / product_train.shape[0])

#%%
regular_customers=product_prior.groupby("order_id")["reordered"].aggregate(sum).reset_index()

regular_customers[regular_customers["reordered"]>1]=1

repurchase_rate=regular_customers["reordered"].value_counts()/regular_customers.shape[0]

"""
repurchasing rate: 0.87947  1代表重複購買的客人  
                   0.12053  0代表初次上門的顧客
"""

#%%
# add to cart order的order是順序的意思，不是訂單，只要查看max就可以知道這一單總共有多少商品！

goods_purchase_number=product_prior.groupby("order_id")["add_to_cart_order"].aggregate(max).reset_index()

purchase_number_count=goods_purchase_number["add_to_cart_order"].value_counts()

plt.figure(figsize=(20,8))

sns.barplot(x=purchase_number_count.index,y=purchase_number_count.values,alpha=0.8)
plt.ylabel("Number of occurence",fontsize=12)
plt.xlabel("Number of products in the given order",fontsize=12)

# rotation --> 設定將x座標字體以垂直方式呈現
plt.xticks(rotation=90)

#%%
# pd.merge() -->合併不同的dataframe欄位

#排列熱銷商品前20名


orders_product_prior=pd.merge(product_prior,products,on="product_id",how="left")
orders_product_prior=pd.merge(orders_product_prior,aisle,on="aisle_id",how="left")
orders_product_prior=pd.merge(orders_product_prior,departments,on="department_id",how="left")

goods_rank=orders_product_prior["product_name"].value_counts().reset_index().head(20)

goods_rank.columns=["Product name","frequency"]


#%%  製作goods_rank 圖表
plt.figure(figsize=(20,8))

sns.barplot(goods_rank["Product name"], goods_rank["frequency"],alpha=0.8,color=color[7])
plt.xlabel("Product name",fontsize=12)
plt.ylabel("Frequency",fontsize=12)
plt.xticks(rotation=270)

#%% 排名aisle欄商品銷量前20
# value_counts(normalize=True)  --> 各個值的相對頻率

aisle_rank=orders_product_prior["aisle"].value_counts().reset_index().head(20)
aisle_rank.columns=["product name","frequency"]

plt.figure(figsize=(20,8))
sns.barplot(x=aisle_rank["product name"],y=aisle_rank["frequency"],alpha=0.7)
plt.xlabel("Product name",fontsize=14)
plt.ylabel("Frequency",fontsize=14)
plt.xticks(rotation=90)

"""
結論 : 可發現主打蔬菜、水果等健康食品

"""


#%%
# 以視覺化方式探討商品放入購物車順序與複購率之關聯

orders_product_prior["add_to_cart_mode"]=orders_product_prior["add_to_cart_order"].copy()
groupby_reorder=orders_product_prior.groupby("add_to_cart_mode")["reordered"].aggregate("mean").reset_index()

groupby_reorder=groupby_reorder.drop(index=69)

#%%

plt.figure(figsize=(20,8))

sns.pointplot(x=groupby_reorder["add_to_cart_mode"],y=groupby_reorder["reordered"].values,alpha=0.8)

plt.xlabel("order put into cart",fontsize=12)
plt.ylabel("Reordered ratio",fontsize=12)

plt.title("Add to cart -- Reordered ratio",fontsize=18)

#%%

# 觀察不同產品的複購率
department_reorder=orders_product_prior.groupby("department")["reordered"].aggregate("mean").reset_index()
department_reorder=department_reorder.drop(index=0)

plt.figure(figsize=(20,8))
sns.pointplot(x=department_reorder["department"],y=department_reorder["reordered"],alpha=0.8,color=color[2])
plt.xlabel("Items",fontsize=12)
plt.ylabel("Reordered ratio",fontsize=12)
plt.title("Items -- Reordered ratio",fontsize=18)
plt.xticks(rotation=90)












