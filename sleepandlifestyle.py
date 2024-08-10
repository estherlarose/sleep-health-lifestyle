# Esther Larose
# Sleep and LifeStyle Data Anaylsis - Python
# August 10 2024

#Reading data
import pandas as pd

#Fixings warnings
import warnings 
warnings.filterwarnings('ignore')

#Mathematical operations
import numpy as np

#Visualisation
import seaborn as sns 
import plotly.express as px
from termcolor import colored
import matplotlib.pyplot as plt 
import plotly.graph_objects as go
import plotly.figure_factory as ff

#Data spliting
from sklearn.model_selection import train_test_split

sleep_data = pd.read_csv('/Users/loserlarose/Desktop/Sleep_health_and_lifestyle_dataset.csv')

#head() for display the first 5 rows  
sleep_data.head().style.set_properties(**{'background-color': '#4A235A',
                                          'color': '#E2EEF3'}) #for colored output

shape = colored(sleep_data.shape, "magenta",None, attrs=["blink"])
print('The dimention of data is :',shape)

# The dimenson of the data is (374,13)

sleep_data.info() # for empty and type of values

#RangeIndex: 374 entries, 0 to 373
#Data columns (total 13 columns):
#   Column                   Non-Null Count  Dtype  
#---  ------                   --------------  -----  
# 0   Person ID                374 non-null    int64  
# 1   Gender                   374 non-null    object 
# 2   Age                      374 non-null    int64  
# 3   Occupation               374 non-null    object 
# 4   Sleep Duration           374 non-null    float64
# 5   Quality of Sleep         374 non-null    int64  
# 6   Physical Activity Level  374 non-null    int64  
# 7   Stress Level             374 non-null    int64  
# 8   BMI Category             374 non-null    object 
# 9   Blood Pressure           374 non-null    object 
# 10  Heart Rate               374 non-null    int64  
# 11  Daily Steps              374 non-null    int64  
# 12  Sleep Disorder           155 non-null    object 
#dtypes: float64(1), int64(7), object(5)

# for statistical info
sleep_data.describe().style.background_gradient(cmap= 'BuPu') #for colored output

#for statistical info including string values
sleep_data.describe(include='O').style.set_properties(**{'background-color': '#4A235A',
                                                      'color': '#E2EEF3'}) 

columns_name=colored(sleep_data.columns, 'magenta',None, attrs=["blink"]) #for show names of columns
print(columns_name)

#Index(['Person ID', 'Gender', 'Age', 'Occupation', 'Sleep Duration',
#       'BMI Category', 'Blood Pressure', 'Heart Rate', 'Daily Steps',
#       'Sleep Disorder'],
#      dtype='object')

plt.style.use('seaborn-v0_8-white')
sns.pairplot(data=sleep_data.drop('Person ID', axis=1), hue='Sleep Disorder', palette='magma')
plt.legend()
plt.show()

# Percentage of person(s) that have sleep disorder or not
classes=colored(sleep_data['Sleep Disorder'].unique(), "magenta",None, attrs=["blink"])
print('The outputs from the classification are :',classes)

# The outputs from the classification are :  [None 'Sleep Apnea' 'Insomnia']

sleep_data['Sleep Disorder'].value_counts()

#It is clear that the proportion of normal people is more

fig=px.histogram(sleep_data,x='Sleep Disorder', 
                 barmode="group",color='Sleep Disorder',
                 color_discrete_sequence=['white','#4A235A','#C39BD3'],
                 text_auto=True)
 
    
fig.update_layout(title='<b>Distribution of persons have sleep disorder or not</b>..',
                 title_font={'size':25},
                 paper_bgcolor='#EBDEF0',
                 plot_bgcolor='#EBDEF0',
                 showlegend=True)


fig.update_yaxes(showgrid=False)
fig.show()

Gender=colored(sleep_data['Gender'].unique(), "magenta",None, attrs=["blink"])
print('The values of Sex column are:', Gender)
#The values of Sex column are: ['Male' 'Female']

sleep_data.groupby('Sleep Disorder') ['Gender'].value_counts()

#Sleep Disorder  Gender
#Insomnia        Male      41
#                Female    36
#Sleep Apnea     Female    67
#                Male      11
#Name: count, dtype: int64

#Observation: It is clear that normal men more than women
# Men suffer from insomnia more than women
# Women suffer from sleep Apnea more than Men


sleep_data.groupby('Sleep Disorder')['Gender'].value_counts().plot.pie(autopct ='%1.1f%%',figsize=(15,7),
                                                                       colors=['#4A235A','pink','#4A235A','pink','#4A235A','pink'])
plt.title('The relationship between (sex) and (Sleep Disorder)')
plt.axis('equal')
plt.show()

jobs=colored(sleep_data['Occupation'].unique(), "magenta", None, attrs=["blink"])
print('The types of jobs that exist are:', jobs)
#The types of jobs that exist are: ['Software Engineer' 'Doctor' 'Sales Representative' 'Teacher' 'Nurse'
#'Engineer' 'Accountant' 'Scientist' 'Lawyer' 'Salesperson' 'Manager']

sleep_data.groupby('Sleep Disorder')['Occupation'].value_counts()
#Sleep Disorder  Occupation          
#Insomnia        Salesperson             29
#                Teacher                 27
#                Accountant               7
#                Engineer                 5
#                Doctor                   3
#                Nurse                    3
#                Lawyer                   2
#                Software Engineer        1
#Sleep Apnea     Nurse                   61
#                Doctor                   4
#                Teacher                  4
#                Lawyer                   3
#                Sales Representative     2
#                Scientist                2
#                Engineer                 1
#                Salesperson              1

#Observation : It is clear that Normal, Doctor more than others
# The people who suffer from insomnia, Salesperson more than others
# The people who suffer from Sleep Apnea, Nurse more than others

sleep_data['Sleep Disorder'].fillna('No Disorder', inplace=True)

fig=px.treemap(sleep_data,path=[px.Constant('Jobs'),'Sleep Disorder','Occupation'],
               color='Sleep Disorder',
              color_discrete_sequence=['#EBDEF0','#C39BD3','#4A235A'])


fig.update_layout(title='<b>The effect of job on sleep</b>..',
                 title_font={'size':20})


fig.show()

sleep_data.pivot_table(index='Quality of Sleep',columns='Sleep Disorder',values='Sleep Duration',aggfunc='mean').style.background_gradient(cmap='BuPu')

fig=px.sunburst(sleep_data,path=[px.Constant('Sleep quality'),'Sleep Disorder','Quality of Sleep'],
               color='Sleep Disorder',values='Sleep Duration',
              color_discrete_sequence=['pink','#4A235A','#FFF3FD'],
              hover_data=['Gender'])

fig.update_layout(title='<b>The effect of quality of sleep on sleep </b>..',
                 title_font={'size':25})

fig.show()

sleep_data.pivot_table(index='Gender',columns='Sleep Disorder',values='Age',aggfunc='median').plot(kind='bar',color={'#FFF3FD','#4A235A','pink'},
                                                                                                   title='Most affected ages in each type of Sleep Disorder',
                                                                                                    label='Age',alpha=.7)


plt.show()

fig=px.ecdf(sleep_data,x='Age',
            color='Sleep Disorder',
            color_discrete_sequence=['white','#4A235A','#C39BD3'])


fig.update_layout(title='<b>The effect of ages on sleep </b>..',
                 title_font={'size':25},
                 paper_bgcolor='#EBDEF0',
                 plot_bgcolor='#EBDEF0')


fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()

fig=px.histogram(sleep_data,x='Sleep Disorder',y='Sleep Duration',
                 color='Sleep Disorder',color_discrete_sequence=['white','#4A235A','#C39BD3'],
                 text_auto=True)



fig.update_layout(title='<b>The effect of Sleep Duration on Sleep Disorder</b> ..',
                  titlefont={'size': 24,'family': 'Serif'},
                  showlegend=True, 
                  paper_bgcolor='#EBDEF0',
                  plot_bgcolor='#EBDEF0')


fig.update_yaxes(showgrid=False)
fig.show()

fig=px.scatter_3d(sleep_data,x='BMI Category',y='Blood Pressure',z='Heart Rate',
                  color='Sleep Disorder',width=1000,height=900,
                  color_discrete_sequence=['white','#4A235A','#C39BD3'])


fig.update_layout(title='<b>The relationship between (BMI Category , Blood Pressure and Heart Rate) and their effect on  Sleep Disorder</b> ..',
                  titlefont={'size': 20,'family': 'Serif'},
                  showlegend=True)


fig.show()

sleep_data.pivot_table(index='Stress Level',columns='Sleep Disorder',aggfunc={'Sleep Disorder':'count'}).style.background_gradient(cmap='BuPu')

fig=px.histogram(sleep_data,x='Sleep Disorder',
                 color='Sleep Disorder',
                 facet_col='Stress Level',
                 barmode='group',
                 color_discrete_sequence=['white','#4A235A','#C39BD3'],
                 opacity=.8)


fig.update_layout(title='<b>The effect of Stress Level on Sleep Disorder</b> ..',title_font={'size':30},
                  paper_bgcolor='#EBDEF0',
                  plot_bgcolor='#EBDEF0')



fig.update_yaxes(showgrid=False)
fig.show()

BMI_Category=colored(sleep_data['BMI Category'].unique(), "magenta",None, attrs=["blink"])
print('The values of BMI Category column are :',BMI_Category)

#The values of BMI Category column are : ['Overweight' 'Normal' 'Obese' 'Normal Weight']

sleep_data.pivot_table(index='BMI Category',columns='Sleep Disorder',aggfunc={'Sleep Disorder':'count'}).style.background_gradient(cmap='BuPu')

sleep_data.pivot_table(index='BMI Category',columns='Sleep Disorder',aggfunc={'Sleep Disorder':'count'}).plot.pie(autopct ='%1.1f%%',
                                                                                                                  subplots=True,figsize=(20,10),
                                                                                                                  colors=['#C39BD3','#D2B4DE','#EBDEF0','#F4ECF7'])

plt.axis('equal')
plt.show()

#Data Reprocessing
sleep_data.isna().sum()

#Person ID                  0
#Gender                     0
#Age                        0
#Occupation                 0
#Sleep Duration             0
#Quality of Sleep           0
#Physical Activity Level    0
#Stress Level               0
#BMI Category               0
#Blood Pressure             0
#Heart Rate                 0
#Daily Steps                0
#Sleep Disorder             0
#dtype: int64

sns.heatmap(sleep_data.isna(),cmap='BuPu')

sleep_data.columns

#Index(['Person ID', 'Gender', 'Age', 'Occupation', 'Sleep Duration',
#       'Quality of Sleep', 'Physical Activity Level', 'Stress Level',
#       'BMI Category', 'Blood Pressure', 'Heart Rate', 'Daily Steps',
#       'Sleep Disorder'],
#      dtype='object')

sleep_data['Blood Pressure'].unique()
#array(['126/83', '125/80', '140/90', '120/80', '132/87', '130/86',
#       '117/76', '118/76', '128/85', '131/86', '128/84', '115/75',
#       '135/88', '129/84', '130/85', '115/78', '119/77', '121/79',
#       '125/82', '135/90', '122/80', '142/92', '140/95', '139/91',
#       '118/75'], dtype=object)

# Ideal blood pressure systolic (upper number) : less than 120 , diastolic (bottom number) : less than 80
# Normal systolic (upper number) : in range (120 - 129) , diastolic (bottom number) : in range (80 - 84)
# Otherwise, blood pressure is high

sleep_data['Blood Pressure']=sleep_data['Blood Pressure'].apply(lambda x:0 if x in ['120/80','126/83','125/80','128/84','129/84','117/76','118/76','115/75','125/82','122/80'] else 1)
# 0 = normal blood pressure
# 1 = abnormal blood pressure

sleep_data["Age"]=pd.cut(sleep_data["Age"],2)
sleep_data["Heart Rate"]=pd.cut(sleep_data["Heart Rate"],4)
sleep_data["Daily Steps"]=pd.cut(sleep_data["Daily Steps"],4)
sleep_data["Sleep Duration"]=pd.cut(sleep_data["Sleep Duration"],3)
sleep_data["Physical Activity Level"]=pd.cut(sleep_data["Physical Activity Level"],4)

from sklearn.preprocessing import LabelEncoder #for converting non-numeric data (String or Boolean) into numbers
LE=LabelEncoder()

categories=['Gender','Age','Occupation','Sleep Duration','Physical Activity Level','BMI Category','Heart Rate','Daily Steps','Sleep Disorder']
for label in categories:
    sleep_data[label]=LE.fit_transform(sleep_data[label])
    
sleep_data.drop(['Person ID'], axis=1, inplace=True)

correlation=sleep_data.corr()
max_6_corr=correlation.nlargest(6,"Sleep Disorder")
sns.heatmap(max_6_corr,annot=True,fmt=".2F",annot_kws={"size":8},linewidths=0.5,cmap='BuPu')
plt.title('Maximum six features affect Sleep Disorder')
plt.show()
