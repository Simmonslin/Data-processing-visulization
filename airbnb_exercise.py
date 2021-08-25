import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode(connected=True) #能夠離線使用

import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
import cufflinks   #連結 python 跟 plotly的軟體


%matplotlib inline
calendar_airbnb=pd.read_csv("calendar_airbnb.csv")
#%%
color=sns.color_palette()

#%%
# nunique() --> 跑出unique()後的數量
listing_id_count=calendar_airbnb["listing_id"].value_counts().reset_index()
listing_id_count.columns=["listing_id","counts"]

print("總計有",calendar_airbnb["date"].nunique(),"天和",calendar_airbnb["listing_id"].nunique(),"種清單")


#%%
print(calendar_airbnb["listing_id"].nunique())
print(calendar_airbnb["date"].min(),calendar_airbnb["date"].max())

#%%  統計可不可入住比率
available_count=calendar_airbnb["available"].value_counts(normalize=True).reset_index()
available_count.columns=["available","counts"]
#%%  統計可不可入住比率 -- 長條圖 
plt.figure(figsize=(12,8))
sns.barplot(x=available_count["available"],y=available_count["counts"],alpha=0.8,color=color[9])
plt.xlabel("Available",fontsize=14)
plt.ylabel("Ratio",fontsize=14)
plt.title("Available ratio",fontsize=18)
plt.xticks(fontsize=16)

#%%  計算熱度分佈並進行視覺化
from matplotlib.font_manager import FontProperties
myfont=FontProperties(fname=r'C:\Users\howar\Desktop\msj.ttf',size=14)
sns.set(font=myfont.get_family())
sns.set_style("whitegrid",{"font.sans-serif":['Microsoft JhengHei']})


new_calendar=calendar_airbnb[["date","available"]]

# map(function, iterable) iterable --> 一個或多個序列
new_calendar["busy"]=new_calendar["available"].map(lambda x:0 if x=="t" else 1)

new_calendar_date_ratio=new_calendar.groupby("date")["busy"].aggregate("mean").reset_index()

#pd.to_datetime(arg, errors='raise', dayfirst=False, yearfirst=False, utc=None, format=None, exact=True, unit=None, infer_datetime_format=False, origin='unix', cache=True)
new_calendar_date_ratio["date"]=pd.to_datetime(new_calendar_date_ratio["date"]) #進行時間排序

plt.figure(figsize=(20,8))
sns.lineplot(x=new_calendar_date_ratio["date"], y=new_calendar_date_ratio["busy"])
plt.xlabel("Date",fontsize=14)
plt.ylabel("Order rate",fontsize=14)
plt.title("訂房率趨勢圖")
#%%
calendar_airbnb["date"]=pd.to_datetime(calendar_airbnb["date"])
# dataframe replace --> df.str.replace
# 一般字串 --> df.replace 
calendar_airbnb["price"]=calendar_airbnb["price"].str.replace(",","").str.replace("$","").astype(float)
calendar_date_price=calendar_airbnb.groupby("date")["price"].aggregate("mean").reset_index()

#%% 計算每月平均房價
# 將date 中的月份取出  -->需使用series.df.strftime()-->取出英文月份
mean_month_price=calendar_airbnb.groupby(calendar_airbnb["date"].dt.strftime("%B"),sort=False)["price"].aggregate("mean")

# 使用pandas.Dataframe.plot 製圖
mean_month_price.plot(figsize=(12,7),kind="barh")
plt.xlabel("月平均房價",fontsize=14)
plt.ylabel("月份",fontsize=14)
plt.title("每月平均房價圖")





#%% 分析airbnb 的 listing 資料

airbnb_listing=pd.read_csv("airbnb_listings.csv")

print(airbnb_listing["id"].nunique(),"個不同的 list")

#%%  哪個區最多空房 ?

airbnb_listing_neighbourhood=airbnb_listing["neighbourhood"].value_counts()
airbnb_listing_neighbourhood.plot(kind="bar",figsize=(20,8),color=color[1])
plt.xticks(rotation=0)
plt.ylabel("空房數",fontsize=14)
plt.title("哪個區最多房源 ?",fontsize=20)

#%% 價格分布情況   sns.distplot() -->單向量分布圖
plt.figure(figsize=(12,8))
sns.distplot(airbnb_listing[airbnb_listing["price"]<50000].price.dropna())
sns.despine()
plt.title("房價大致分布",fontsize=15)

# .describe() --> 列出各項統計數據
print(airbnb_listing[airbnb_listing["price"]<50000].price.describe())

#%%
airbnb_max_price=airbnb_listing[airbnb_listing["price"]>290000]
airbnb_max_price=airbnb_max_price.append(airbnb_listing[airbnb_listing["price"]<100])

#%%
plt.figure(figsize=(12,7))
airbnb_listing[(airbnb_listing["price"]<10000) & (airbnb_listing["price"]>300)].price.hist(bins=20)
plt.title("便宜實惠的房價",fontsize=15)
plt.xlabel("Listing Price",fontsize=14)
plt.ylabel("Count",fontsize=14)

#%% 製作不同區跟房價之間關係的盒型圖  boxplot( x="",y="",data=,order= )
rational_price_range=airbnb_listing[(airbnb_listing["price"]<=10000) &(airbnb_listing["price"]>=300)]

sort_price=rational_price_range.groupby("neighbourhood")["price"].median().sort_values(ascending=False).index
plt.figure(figsize=(16,6))

sns.boxplot(x="neighbourhood",y="price",data=rational_price_range,order=sort_price)

#%% 房間類型與價格關係盒形圖

sort_price_room=rational_price_range.groupby("room_type")["price"].median().sort_values(ascending=False).index
plt.figure(figsize=(16,6))
sns.boxplot(x="room_type",y="price",data=rational_price_range,order=sort_price_room)
plt.xlabel("房間類型",fontsize=14)
plt.ylabel("價格",fontsize=14)
plt.title("房間類型屬性影響",fontsize=16)
#%%

def sort_price(num_1,num_2,label_1,label_2):
    price_range=airbnb_listing[(airbnb_listing["price"]<=num_1) & (airbnb_listing["price"]>=num_2)]
    sort_price=price_range.groupby(label_1)[label_2].median().sort_values(ascending=False).index
    plt.figure(figsize=(90,6))
    sns.boxplot(x=label_1,y=label_2,data=airbnb_listing,order=sort_price)
    
sort_price(10000,300,"host_name","price")

#%%

hotel_value_counts=airbnb_listing["host_name"].value_counts().reset_index()
hotel_value_counts.columns=["host_name","counts"]





    






 





