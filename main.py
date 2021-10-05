import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
df = pd.read_csv('F:/2020/Monthly Segregated/January/BANKNIFTY.csv')
df['TIME'] = pd.to_datetime(df['TIME']).dt.time
df['DATE'] = pd.to_datetime(df["DATE"])
df['Day'] = pd.DatetimeIndex(df['DATE']).day
df1 = df.drop(['NAME','HIGH','Unnamed: 7','Unnamed: 8','DATE','TIME'],axis=1)
df2 = df['HIGH']
x_train,x_test,y_train,y_test = train_test_split(df1,df2,test_size=0.9,random_state=1)
reg = LinearRegression()
reg.fit(x_train,y_train)
prediction = reg.predict(x_test)
prediction = pd.DataFrame(prediction)
prediction['day'] = df['Day']
prediction.rename( columns={0:'predicted'}, inplace=True )
sns.relplot(data=df, x="Day", y="HIGH", kind="line")
sns.relplot(data=prediction,x="day",y= 'predicted',kind="line")
