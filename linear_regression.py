def fit(x_train,y_train):
    #gives tuple (slope,intercept) as output where m and c const for best fitting line
    m=((x_train*y_train).mean()-x_train.mean()*y_train.mean())/((x_train**2).mean()-x_train.mean()**2)
    c=y_train.mean()-m*x_train.mean()
    return m,c
def predict(input,m,c):
    #takes input and m,c of fitting line as parameter and return predicted output
    return input*m+c
def coefficient_of_determination(y_predicted,y_test):
    u=((y_predicted-y_test)**2).sum() #sum of individual corresponding difference squares
    v=((y_test-y_test.mean())**2).sum()
    return 1-(u/v)
def cost(x_test,y_test,m,c):
    return ((y_test-(m*x_test+c))**2).mean()
import numpy as np
data=np.loadtxt("data.csv",delimiter=",")
input=data[:,0]
output=data[:,1]
#so input has all the input data and output has all the output datas
#convert to numpy array as mean wagera nikalna easier
input=np.array(input)
output=np.array(output)
from sklearn import model_selection
training_input,testing_input,training_output,testing_output=model_selection.train_test_split(input,output)
#so random datas when to all 4 categories
#np array ko split to 4 parts toh woh 4 parts bhi np array hi
m,c=fit(training_input,training_output)
#print(m)
#print(c)
#got m and c from fit function
#print(predict(testing_input[3],m,c)," ",testing_output[3]) so aise checked bhi ki test mei 3rd datapoint ka ans kitna close
#print(coefficient_of_determination(m*testing_input+c,testing_output))#as m multiplied by all elements in array and c added to all gives all corresponding y predicted
print("Slope : ",m,"\nIntercept : ",c)
print("Coefficient of determination/Score : ",coefficient_of_determination(predict(testing_input,m,c),testing_output))
print("Cost/Average error per data point : ",cost(testing_input,testing_output,m,c))