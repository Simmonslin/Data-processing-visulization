import pandas as pd


diamonds_data=pd.read_csv("diamonds.csv")
diamonds_data_copy=diamonds_data
diamonds_data=diamonds_data.drop(diamonds_data.columns[8:11],axis=1)
diamonds_data=diamonds_data.drop(diamonds_data.columns[5:7],axis=1)



#%%

def carat_selection(a,b,number):
    carat_selection=diamonds_data[diamonds_data["carat"]>=a]
    carat_selection=carat_selection[carat_selection["carat"]<=b]
    carat_selection=carat_selection.head(number)
    carat_selection=carat_selection.sort_values(by="color",ascending=True)
    return carat_selection

carat_24=carat_selection(0,0.49,10)
carat_57=carat_selection(0.5,0.99,10)
carat_8_10=carat_selection(1,1.49,10)
carat_11_13=carat_selection(1.5,1.99,10)
carat_14_16=carat_selection(2,2.49,10)
carat_over2=carat_selection(2.5,3,10)

#%% 合併資料

diamonds_organized=carat_24.append([carat_57,carat_8_10,carat_11_13,carat_14_16,carat_over2])

#%%  將值排序
diamonds_organized=diamonds_organized.sort_values(by="carat",ascending=True)

#%% 計算刀工等級分佈
diamonds_cut_count=diamonds_organized.loc[:,"cut"].value_counts()

#%% 重置index
#inplace=True  --> 將index重製成預設索引
# drop = True --> 初始索引從 DataFrame中刪除
# drop = False --> 初始索引留在 DataFrame 中,自成一列

diamonds_organized.reset_index(level=None,
                      drop=True, 
                      inplace=True, 
                      col_level=0,
                       col_fill='')


#%% 
"""
將 cut數字化 

level: fair --> Ideal --> Good --> Very Good --> Premium

        1         2        3           4           5
"""

diamonds_organized.iloc[diamonds_organized[diamonds_organized["cut"]=="Premium"].index,2]=5
diamonds_organized.iloc[diamonds_organized[diamonds_organized["cut"]=="Very Good"].index,2]=4
diamonds_organized.iloc[diamonds_organized[diamonds_organized["cut"]=="Good"].index,2]=3
diamonds_organized.iloc[diamonds_organized[diamonds_organized["cut"]=="Ideal"].index,2]=2
diamonds_organized.iloc[diamonds_organized[diamonds_organized["cut"]=="Fair"].index,2]=1

#%% 製作平均價格 DataFrame


price_mean=pd.DataFrame(columns=["0.2-0.4","0.5-0,7","0.8-1","1.1-1.3","1.4-1.6","2-3"],
                        data=[[740,1890,4185,6670,10380,14837]])

cut_level=pd.DataFrame(data=[[1,2,3,4,5]],
                       columns=["Fair","Ideal","Good","Very Good","Premium"])

#%%算出不同克拉層級的平均價格


def carat_selection_price(a,b):
    carat_selection=diamonds_data[diamonds_data["carat"]>=a]
    carat_selection=carat_selection[carat_selection["carat"]<=b]
    return carat_selection["price"].mean()

print(carat_selection_price(0,0.49))
print(carat_selection_price(0.5,0.99))
print(carat_selection_price(1,1.49))
print(carat_selection_price(1.5,1.99))
print(carat_selection_price(2,2.49))
print(carat_selection_price(2.5,3))

#%%
diamonds_organized=diamonds_organized.drop(labels=["price",'Unnamed: 0'],axis=1)

#%% 將變數轉換成excel檔

with pd.ExcelWriter("output.xlsx") as writer:
    diamonds_organized.to_excel('output1.xlsx', engine='xlsxwriter')
    diamonds_organized.to_excel(writer,"作業研究期末數據")

#%%

# 第1欄 : 取檔案名

price_mean.to_excel("各克拉層級平均價格.xlsx",sheet_name="price_mean")
cut_level.to_excel("切工等級代號.xlsx", sheet_name="cut_level")

#%%
diamonds_organized.to_excel("作業研究期末數據_修改.xlsx",sheet_name="management_science")
#%%
# 隨機抽取資料
def carat_selection_random(a,b,number):
    carat_selection=diamonds_data[diamonds_data["carat"]>=a]
    carat_selection=carat_selection[carat_selection["carat"]<=b]
    carat_random=carat_selection.sample(n=number).reset_index()
    carat_random=carat_random.drop(columns=["index",'Unnamed: 0'])
    carat_random=carat_random.sort_values(by=["carat"],ascending=True)
    return carat_random

carat_49_random=carat_selection_random(0,0.49,10)
carat_510_random=carat_selection_random(0.5,0.99,10)
#%%
print(carat_49_random.columns)


    



    












    







