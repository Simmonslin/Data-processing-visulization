import pandas as pd 
import seaborn as sns
import numpy as np

telecom_customer=pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 將 dataframe 文字部分轉成 float
telecom_customer["TotalCharges"]=pd.to_numeric(telecom_customer["TotalCharges"].str.replace(" ",""))

genders_charge=telecom_customer.groupby("gender")["TotalCharges"].aggregate("mean")


#%%
print(telecom_customer.columns)

#%%
print(type(telecom_customer.iloc[0,19]))

#%% 找出缺失值數量
print(telecom_customer.isnull().sum())
print(telecom_customer.nunique())

#%% 從分佈圖發現數據可能有極端值
sns.distplot(telecom_customer["tenure"])

# 檢查是否存有異常值, 若值大於0.8代表有極端值
def check_bad_smell(df):
    error_event =abs((df.mode().iloc[0,] - df.median())/df.quantile(0.75) - df.quantile(0.25))
    problems = error_event[error_event>0.8]
    print(problems)
    return problems.index.tolist()

bad_smell=check_bad_smell(telecom_customer)

#%%  替代掉空白值當作null
telecom_customer["TotalCharges"]=telecom_customer["TotalCharges"].replace(" ",np.nan)
print(telecom_customer["TotalCharges"].isnull().sum())

print(telecom_customer[telecom_customer["TotalCharges"].isnull()])

# 排除具null值的資料

telecom_customer=telecom_customer[telecom_customer["TotalCharges"].notnull()]

telecom_customer["TotalCharges"]=telecom_customer["TotalCharges"].astype(float)

#%%  將 YES / No 轉化成 0 和 1
#  --> Series: Series.str.replace()  整個DataFrame --> Dataframe.replace(old,new)
telecom_customer=telecom_customer.replace(["Yes","No"],[1,0])

# 另外需處理 no phone service 的資料
telecom_customer=telecom_customer.replace("No phone service",0)

print(telecom_customer["tenure"].describe())


#%%  將 tenure 轉變為離散變量 (數值只能用自然數 or 整數單位計算)  --> 將 tenure 劃分成5個等級
# --> pd.cut()

def tenure_to_bins (df):
    labels=[1,2,3,4,5]
    bins=pd.cut(df,bins=5,labels=labels)
    return bins 

telecom_customer["tenure_level"]=tenure_to_bins(telecom_customer["tenure"])

# 更改 columns順序 --> 使用dataframe.reindex() or dataframe.reset_index()
telecom_customer=telecom_customer.reindex(columns=['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'tenure',"tenure_level", 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn',])



#%%
import plotly.offline as py
py.init_notebook_mode(connected=True) #為了能在本地端調用
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff

#%%  使用 go 製作圓餅圖

# 將churn 分群後依 keys 和 values 分成2個 list  Churn ---> 顧客流失
churn_keys=telecom_customer["Churn"].value_counts().keys().tolist()
churn_values=telecom_customer["Churn"].value_counts().values.tolist()

churn_pie=go.Pie(labels=churn_keys,values=churn_values,marker=dict(colors=["royalblue","lime"],
                                                                   line=dict(color="white",width=1.3),
                                                                   ),rotation=90,
            # 懸停資訊      
            hoverinfo="label+value+text", hole=.5 # 設定空心大小 
            )

# 使用 go.Layout() : 添加 axis, legend(图例), margin（旁注）, paper 和 plot properties

churn_layout=go.Layout(title="顧客流失資料",plot_bgcolor="rgb(243,243,243)",paper_bgcolor="rgb(243,243,243)"
                       )

## 結合數據與介面
figure=go.Figure(data=[churn_pie],layout=churn_layout)

# py.iplot()  --> Jupyter Notebook   py.plot() --> Spyder
py.plot(figure)

"""
結論 :  客戶流失、信用欺詐等數據分析專案都是所謂的「不平衡資料集」，也就是說流失屬於「稀少事件」

在大型數據（幾億筆資料）會用抽樣方法來平衡兩者

"""


                                                                   
#%%
#  --> annotation (x= float,y= float) --> 設定x,y註解的位置

def plot_pie(churn,not_churn,column,position_x1,position_y1,position_x2,position_y2) :
    
    trace1 = go.Pie(values  = churn[column].value_counts().values.tolist(),
                    labels  = churn[column].value_counts().keys().tolist(),
                    hoverinfo = "label+percent+name",
                    domain  = dict(x = [0,.48]),
                    name    = "Churn Customers",
                    marker  = dict(line = dict(width = 2,
                                               color = "rgb(243,243,243)")
                                  ),
                    hole    = .6
                   )
    trace2 = go.Pie(values  = not_churn[column].value_counts().values.tolist(),
                    labels  = not_churn[column].value_counts().keys().tolist(),
                    hoverinfo = "label+percent+name",
                    marker  = dict(line = dict(width = 2,
                                               color = "rgb(243,243,243)")
                                  ),
                    domain  = dict(x = [.52,1]),
                    hole    = .6,
                    name    = "Non churn customers" 
                   )
    layout = go.Layout(dict(title = column + " distribution in customer attrition ",
                            plot_bgcolor  = "rgb(243,243,243)",
                            paper_bgcolor = "rgb(243,243,243)",
                            annotations = [dict(text = "churn customers",
                                                font = dict(size = 13),
                                                showarrow = False,
                                                x = position_x1, y = position_y1),
                                           dict(text = "Non churn customers",
                                                font = dict(size = 13),
                                                showarrow = False,
                                                x = position_x2,y = position_y2
                                               )
                                          ]
                           )
                      )
    data = [trace1,trace2]
    fig  = go.Figure(data = data,layout = layout)
    py.plot(fig)
#function  for histogram for customer attrition types
def plot_hist(churn,not_churn,column) :
    trace1 = go.Histogram(x  = churn[column],
                          histnorm= "percent",
                          name = "Churn Customers",
                          marker = dict(line = dict(width = .5,
                                                    color = "black"
                                                    )
                                        ),
                         opacity = .9 
                         ) 
    
    trace2 = go.Histogram(x  = not_churn[column],
                          histnorm = "percent",
                          name = "Non churn customers",
                          marker = dict(line = dict(width = .5,
                                              color = "black"
                                             )
                                 ),
                          opacity = .9
                         )
    
    data = [trace1,trace2]
    layout = go.Layout(dict(title =column + " distribution in customer attrition ",
                            plot_bgcolor  = "rgb(243,243,243)",
                            paper_bgcolor = "rgb(243,243,243)",
                            xaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                             title = column,
                                             zerolinewidth=1,
                                             ticklen=5,
                                             gridwidth=2
                                            ),
                            yaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                             title = "percent",
                                             zerolinewidth=1,
                                             ticklen=5,
                                             gridwidth=2
                                            ),
                           )
                      )
    fig  = go.Figure(data=data,layout=layout)
    
    py.plot(fig)
    
#function  for scatter plot matrix  for numerical columns in data
def plot_scatter(df)  :
    
    df  = df.sort_values(by = "Churn" ,ascending = True)
    classes = df["Churn"].unique().tolist()
    classes
    
    class_code  = {classes[k] : k for k in range(2)}
    class_code
    color_vals = [class_code[cl] for cl in df["Churn"]]
    color_vals
    pl_colorscale = "Portland"
    pl_colorscale
    text = [df.loc[1:7032,"Churn"] ]
   
    trace = go.Splom(dimensions = [dict(label  = "tenure",
                                       values = df["tenure"]),
                                  dict(label  = 'MonthlyCharges',
                                       values = df['MonthlyCharges']),
                                  dict(label  = 'TotalCharges',
                                       values = df['TotalCharges'])],
                     text=text,

                     marker = dict(color = color_vals,
                                   colorscale = pl_colorscale,
                                   size = 3,
                                   showscale = False,
                                   line = dict(width = .1,
                                               color='rgb(230,230,230)'
                                              )
                                  )
                    )
    axis = dict(showline  = True,
                zeroline  = False,
                gridcolor = "#fff",
                ticklen   = 4
               )
    
    layout = go.Layout(dict(title  = 
                            "Scatter plot matrix for Numerical columns for customer attrition",
                            autosize = False,
                            height = 800,
                            width  = 800,
                            dragmode = "select",
                            hovermode = "closest",
                            plot_bgcolor  = 'rgba(240,240,240, 0.95)',
                            xaxis1 = dict(axis),
                            yaxis1 = dict(axis),
                            xaxis2 = dict(axis),
                            yaxis2 = dict(axis),
                            xaxis3 = dict(axis),
                            yaxis3 = dict(axis),
                           )
                      )
    data   = [trace]
    fig = go.Figure(data = data,layout = layout )
    py.plot(fig)
    
#%%  plot_pie () 製作圓餅圖

plot_pie(telecom_customer[telecom_customer["Churn"]==1],
         telecom_customer[telecom_customer["Churn"]==0],"gender",
         .19,.5,.83,.5)

#%% 製作tenure - churn 分布圖  (百分比)

plot_hist(telecom_customer[telecom_customer["Churn"]==1],telecom_customer[telecom_customer["Churn"]==0],"tenure")

#%%  tenure_level - churn 分布 (count)
import matplotlib.pyplot as plt
plt.figure(figsize=(16,6))

# 長條圖 --> 1刻度1條  (x=, y=, data=)   1刻度多條  (x=, hue=, data=)
sns.countplot(x="tenure_level",hue="Churn",data=telecom_customer)
# .legend() -->顯示數據名稱
plt.legend(["Non Churn","Churn"])
#%%
# 以三維概念製作二維圖

plot_scatter(telecom_customer)



#%%

df  = telecom_customer.sort_values(by = "Churn" ,ascending = True)
classes = df["Churn"].unique().tolist()

    
class_code  = {classes[k] : k for k in range(2)}
color_vals = [class_code[cl] for cl in df["Churn"]]
    

text = [df.loc[k,"Churn"] for k in range(len(df))]

#%% plot_scatter() --> 製作立體三維圖 -->散點圖
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig=plt.figure()
ax=Axes3D(fig)

ax.scatter(telecom_customer[telecom_customer["Churn"]==0].tenure,telecom_customer[telecom_customer["Churn"]==0].MonthlyCharges,telecom_customer[telecom_customer["Churn"]==0].TotalCharges,c="y",label="Not Churn",alpha=0.1)
ax.scatter(telecom_customer[telecom_customer["Churn"]==1].tenure,telecom_customer[telecom_customer["Churn"]==1].MonthlyCharges,telecom_customer[telecom_customer["Churn"]==1].TotalCharges,c="g",label="Churn",alpha=0.2)
ax.legend()

ax.set_xlabel("Tenure",fontsize=12)
ax.set_ylabel("Monthly Charges",fontsize=12)
ax.set_zlabel("Total Charges",fontsize=12)

#%%  函式化 --> 三圍散點圖

def three_demension_scatter(df,column_name,column_name_2,column_name_3,trait_1,trait_2,color_1,color_2,label_1,label_2):
    fig=plt.figure()
    ax=Axes3D(fig)
    df_series_1=df[df[label_2]==trait_1][column_name]
    df_series_2=df[df[label_2]==trait_1][column_name_2]
    df_series_3=df[df[label_2]==trait_1][column_name_3]
    df2_series_1=df[df[label_2]==trait_2][column_name]
    df2_series_2=df[df[label_2]==trait_2][column_name_2]
    df2_series_3=df[df[label_2]==trait_2][column_name_3]
    ax.scatter(df_series_1,df_series_2,df_series_3,c=color_1,label=label_1,alpha=0.1)
    ax.scatter(df2_series_1,df2_series_2,df2_series_3,c=color_2,label=label_2,alpha=0.2)   
    ax.legend()
 
    ax.set_xlabel(column_name,fontsize=12)
    ax.set_ylabel(column_name_2,fontsize=12)
    ax.set_zlabel(column_name_3,fontsize=12)
 

three_demension_scatter(telecom_customer,"tenure","MonthlyCharges","TotalCharges",0,1,"y","g","Not Churn","Churn")

    
    

#%%  Data Modeling -->數據建模   使用sklearn

# 二元變數 (eg: "Churn" 有 0 和 1)
bin_cols=telecom_customer.nunique()[telecom_customer.nunique()==2].keys().tolist()


multi_cols=telecom_customer.nunique()[telecom_customer.nunique()>2]
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

#  .cat：用於分類數據（Categorical data） .str: 用於字符數據 (String Object data)  .dt: 用於時間數據（datetime-like data）

# 將二元數據編碼
cato=telecom_customer["tenure_level"].cat.codes
labelEncoder=LabelEncoder()



#%%  製作雷達圖

import plotly.graph_objs as go

bin_cols_Churn_0=telecom_customer[telecom_customer["Churn"]==0].loc[:,bin_cols]
bin_cols_Churn_1=telecom_customer[telecom_customer["Churn"]==1].loc[:,bin_cols]

#%%

def gender_label(df,label,old,new):
    df[label]=df[label].str.replace(old,new)
gender_label(bin_cols_Churn_0,"gender","Male","0")   
gender_label(bin_cols_Churn_0,"gender","Female","1")
gender_label(bin_cols_Churn_1,"gender","Male","0")
gender_label(bin_cols_Churn_1,"gender","Female","1")

bin_cols_Churn_0["gender"]=pd.to_numeric(bin_cols_Churn_0["gender"])
bin_cols_Churn_1["gender"]=pd.to_numeric(bin_cols_Churn_1["gender"])


def rader_graph(df_1,df_2,title):
    
    fig=go.Figure()

    fig.add_trace(go.Scatterpolar(theta=df_1.columns.tolist(),
                                  r=[df_1["gender"].sum(),df_1["SeniorCitizen"].sum(),
                                     df_1["Partner"].sum(),df_1["Dependents"].sum(),
                                     df_1["PhoneService"].sum(),df_1["MultipleLines"].sum(),
                                     df_1["PaperlessBilling"].sum()],
                                  fill="toself",
                                  name="Not Churn"))
    fig.add_trace(go.Scatterpolar(theta=bin_cols_Churn_1.columns.tolist(),
                                  r=[df_2["gender"].sum(),df_2["SeniorCitizen"].sum(),
                                     df_2["Partner"].sum(),df_2["Dependents"].sum(),
                                     df_2["PhoneService"].sum(),df_2["MultipleLines"].sum(),
                                     df_2["PaperlessBilling"].sum()],
                                  fill="toself",
                                  name="Churn"))
    
    fig.update_layout(polar=dict(radialaxis=dict(visible=True,
                                                showline=True,
                                                tickwidth=2,
                                                gridcolor="white",
                                                )),title=title,showlegend=True)
    
    py.plot(fig)
    
rader_graph(bin_cols_Churn_0,bin_cols_Churn_1,"Churn-to-Customer")



#%%

print(bin_cols_Churn_0.columns)
#%%

import plotly.express as px
df = px.data.wind()
fig = px.line_polar(df, r="frequency", theta="direction", color="strength", line_close=True,
                    color_discrete_sequence=px.colors.sequential.Plasma_r,
                    template="plotly_dark",)
py.plot(fig)

#%%

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

lb=LabelEncoder()

onehot=OneHotEncoder()

telecom_customer_copy=telecom_customer.copy()
telecom_customer_copy["gender"]=lb.fit_transform(telecom_customer_copy["gender"])



#%%
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

lb=LabelEncoder()

telecom_customer_copy=pd.get_dummies(data=telecom_customer,columns=multi_cols.index.tolist())


#%%
std=StandardScaler()

list_1=multi_cols.index.tolist()

for i in range(len(bin_cols)):
    list_1.append(bin_cols[i])
    
scaled=std.fit_transform(telecom_customer_copy)
scaled=pd.DataFrame(data=scaled)

#%%

print(multi_cols.index.tolist())


























