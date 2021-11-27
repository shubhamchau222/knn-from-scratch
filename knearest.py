from collections import Counter
import numpy as np 


class knn_classifier:
    '''
        Knn-classifier : for 2*2 matrix 
        to get prediction on multiple records : 
                run predict function in for loop : 
                 for example : 
                     predictions = [] 
                    for i in range(x_test.shape[0]):
                        pred = algo.predict(x_test[i])
                        prediction.append(pred)  
    '''
    def __init__(self , k:int=3 ) -> None:
        self.k = k 
        self.X = None
        self.y = None 

    def fit(self,X,y):
        '''
            X : X_train 
            y : y_train        
        '''
        # as we know knn never learn anything hence we store input data into the variables 
        self.X = X
        self.y = y
        print('Training Completed.....')


    def predict(self, x_test):
        # find the distance of Given data points with other data points 
        # find the k shortest distance points 
        # find out there classes & find there mode 

        '''
        X_train : 
        [-0.80276277  0.44295604]
        [-0.70800656  1.43671337]
        
        X_test : [ 
                    [-0.80276277  0.44295604]
                ]
            
            # ( ( x2-x1)^2 + (y2-y1)**2 )**1/2
        '''
        distances = {}
        counter = 0 
        for i in self.X:
            distances[counter] = ((x_test[0][0] - i[0])**2 +  (x_test[0][1] - i[1])**2)**1/2
            counter += 1
        distances = sorted(distances.items() , key=lambda x : x[1] , reverse=False)
        outcomes = []
        for i in range(self.k):
            outcomes.append(self.y[distances[i][0]])   # class available at this index
        
        # let's find the most frequent class 
        most_freq_element = np.bincount(np.array(outcomes)).argmax()
        return most_freq_element

       
            
            
           

    
            



    

        