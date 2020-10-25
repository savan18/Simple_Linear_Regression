
########Simple Linear Regression################
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("salary_Data.csv")
x=data.iloc[:, :-1].values   #IV
y=data.iloc[:, 1].values    #DV

#splitting dataset
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=1/3,random_state=0)

'''#feature Scalling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)'''


#Fitting model
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)

# predict 
y_pred=model.predict(x_test)

#plot the graph
plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,model.predict(x_train))
plt.xlabel("year of experiance")
plt.ylabel("salalry")
plt.title("salary vs experiance of training set")
plt.show()


#testing set graph

plt.scatter(x_test,y_test,color="red")
plt.plot(x_train,model.predict(x_train))
plt.xlabel("year of experiance")
plt.ylabel("salalry")
plt.title("salary vs experiance of test set")
plt.show()



