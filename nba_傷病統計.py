import pandas as pd
injury=pd.read_csv("nba_injuries_2010-2020.csv")

# 先將DATE 從字串轉成datetime格式，再取年份
injury["Date"]=pd.to_datetime(injury["Date"]).dt.strftime("%Y")

not_null_acquire=injury[injury["Acquired"].notnull()]
not_null_relinquish=injury[injury["Relinquished"].notnull()]



#%%
injury_player=injury["Relinquished"].value_counts()
injury_team=injury["Team"].value_counts()

#%%  統計各年份傷例總數
injury_year=injury["Date"].value_counts().sort_index(ascending=True)
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = "Yu Gothic" # 游黑體

color=sns.color_palette()
plt.figure(figsize=(10,8))
plt.xlabel("Year",fontsize=13)
plt.ylabel("Counts",fontsize=13)
plt.title("2010-2020傷病統計",fontsize=15)
sns.barplot(x=injury_year.index,y=injury_year.values,alpha=0.7,color=color[2])

#%%
import numpy as np
injury_notes=np.unique(injury["Notes"])
print(injury_notes[864:866])

#%%

# 整理同類型受傷原因
def str_replace(str_1,str_2):
    injury["Notes"][injury["Notes"].str.contains(str_1)]=str_2

injury["Notes"]=injury["Notes"].str.lower()
str_replace("sprained ankle","sprained ankle")
str_replace("sprained right ankle","sprained ankle")
str_replace("sprained left ankle","sprained ankle")
str_replace("illness","illness")
str_replace("rest","rest")
str_replace("sore knee","sore knee")
str_replace("sore left knee","sore knee")
str_replace("sore right knee","sore knee")
str_replace("back spasm","back spasm")
str_replace("concussion","concussion")


injury_note_count=injury["Notes"][injury["Relinquished"].notnull()].value_counts().reset_index()
injury_note_count.columns=['Reasons', 'Notes']
injury_note_count=injury_note_count[injury_note_count["Notes"]>=100]

#%%  繪製76ers 近十年傷病總次數 線性迴歸圖

player_injury_reason=injury.groupby("Relinquished")["Notes"].value_counts()
team_date_injury_numbers=injury.groupby("Team")["Date"].value_counts().unstack()

team_76ers_injury=pd.DataFrame(team_date_injury_numbers.loc["76ers",:]).reset_index()

# 需將 date 時間進行轉換 --> 最後 type 需轉回int 否則會出現Error
team_76ers_injury["Date"]=pd.to_datetime(team_76ers_injury["Date"]).dt.strftime("%Y").astype(int)

team_76ers_injury.columns=["Date","total"]
plt.figure(figsize=(16,6))
sns.lmplot(x="Date", y="total", data=team_76ers_injury)

#%%  製作球員 - 年份 - 傷病次數 三維圖

injury_76ers=injury[(injury["Team"]=="76ers") & (injury["Relinquished"].notnull())].reset_index(drop=True)
injury_76ers.index.name = None
injury_76ers["Date"]=injury_76ers["Date"].astype(int)
injury_76ers=injury_76ers.drop(columns=["Acquired","Notes","Team"])
injury_76ers_copy=injury_76ers.copy()
injury_76ers=pd.DataFrame(injury_76ers.groupby("Date")["Relinquished"].aggregate("value_counts"))

# 
injury_76ers=pd.read_csv("sixers_injury.csv")



#%%

injury_76ers_sort=injury_76ers.sort_values(by=["Total"],ascending=False)

def sixers_player_injury(str_1):
    print(injury_76ers[injury_76ers['Relinquished']==str_1])
    print("\t")


sixers_player_injury("Spencer Hawes")
sixers_player_injury("Joel Embiid")
sixers_player_injury("Ben Simmons")
sixers_player_injury("Dario Saric")
sixers_player_injury("T.J. McConnell")
sixers_player_injury("Robert Covington")
sixers_player_injury("Shake Milton")

axes3d_sixers=injury_76ers_sort[(injury_76ers_sort["Relinquished"]=="Joel Embiid") |(injury_76ers_sort["Relinquished"]=="Furkan Korkmaz")|(injury_76ers_sort["Relinquished"]=="Ben Simmons")]

def f(x):
    if x=="Joel Embiid":
        return 1
    elif x=="Furkan Korkmaz":
        return 2
    else:
        return 3
    
#%%  使用 lambda 修改 Relinquished 欄位 --> 可直接使用 if else or 先寫成函式

axes3d_sixers["Relinquished"]=axes3d_sixers["Relinquished"].apply(lambda x: 1 if x=="Joel Embiid" else(2 if x=="Furkan Korkmaz" else 3))
#%%

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



fig=plt.figure()
ax=Axes3D(fig)

# c= 設定顏色 ,s= 設定散點大小 (matplotlib plot 中的方法)
ax.scatter(axes3d_sixers[axes3d_sixers["Relinquished"]==1].Date,axes3d_sixers[axes3d_sixers["Relinquished"]==1].Relinquished,axes3d_sixers[axes3d_sixers["Relinquished"]==1].Total,label="Joel Embiid",c="m",s=100)
ax.scatter(axes3d_sixers[axes3d_sixers["Relinquished"]==2].Date,axes3d_sixers[axes3d_sixers["Relinquished"]==2].Relinquished,axes3d_sixers[axes3d_sixers["Relinquished"]==2].Total,label= "Furkan Korkmaz",c="c",s=100)
ax.scatter(axes3d_sixers[axes3d_sixers["Relinquished"]==3].Date,axes3d_sixers[axes3d_sixers["Relinquished"]==3].Relinquished,axes3d_sixers[axes3d_sixers["Relinquished"]==3].Total,label="Ben Simmons",c="y",s=100)                   

ax.legend()  
ax.set_xlabel("Date",fontsize=15)
ax.set_ylabel("Player",fontsize=15)
ax.set_zlabel("Injured times",fontsize=15)                                                                                                            
#%%

print(injury_76ers_sort[(injury_76ers_sort["Relinquished"]=="Joel Embiid") |(injury_76ers_sort["Relinquished"]=="Furkan Korkmaz")|(injury_76ers_sort["Relinquished"]=="Ben Simmons")])



#%%































    
    
    

