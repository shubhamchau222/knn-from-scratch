import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from knearest import knn_classifier
from knn_best import Knn


'''
# Table will look like this 
    User ID  Gender  Age  EstimatedSalary  Purchased
0  15624510    Male   19            19000          0
1  15810944    Male   35            20000          0
2  15668575  Female   26            43000          0
3  15603246  Female   27            57000          0
4  15804002    Male   19            76000          0

'''

csv_file = 'Social_Network_Ads.csv'
data = pd.read_csv(csv_file)

X = data.iloc[: , 2:4]
y = data.iloc[:,-1]       # only last column 


# data deviding train & test 
X_train , X_test , y_train , y_test = train_test_split(X,y , test_size=0.2 , random_state=20)

# data preprocessing ( standard Scaling )
Scaler = StandardScaler()
X_train = Scaler.fit_transform(X_train)
X_test = Scaler.transform(X_test)
# print(X_train[:5])

'''
    # scaled data matrixes look like 

    [[-0.80276277  0.44295604]
    [-0.70800656  1.43671337]
    [-0.23422551 -0.5508013 ]
    [ 0.90284902  1.16568865]
    [-1.0870314   0.4730699 ]]

'''
# function to get clear idea about prediction 
def get_output(model_obj , test_array):
    result = model_obj.predict(test_array)
    if result ==  0 :
        print(' not able to purchasee')
    elif result == 1 :
        print('This man is able to purchase product....')
# object of the knn classifier class 

a = Knn(k=7)
a.fit(X_train,y_train)
a.predict(X_test[0])  

'''
(250, 0.0016362729034470635)
(253, 0.02463159277991919)
(219, 0.03626555584862264)
(109, 0.044024185910826394)
(259, 0.05444674639160464)
(231, 0.05873192282251449)
(267, 0.06044240489669139)



'''







