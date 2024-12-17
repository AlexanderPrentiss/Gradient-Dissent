import pandas as pd
import matplotlib.pyplot as plt

class Model:
    def __init__(self, file, start_weight = [0]*24, a = 10**-8): # Spec 1 get data from csv and get data for spec 2
        self.data = pd.read_csv(file)
        self.size = len(self.data)
        self.mean = self.data['Price'].mean()
        self.min = self.data['Price'].min()
        self.max = self.data['Price'].max()
        self.stdev = self.data['Price'].std()
        self.weight = start_weight # optional input for weight and learning coef to allow for training / predicting 
        self.learning_rate = a

    def __str__(self): # str to get basic info about the dataset
        return f'Number of records in training set {self.size}\nMean value of price {self.mean}\nMin: {self.min}\nMax: {self.max}\nStandard deviation: {self.stdev}'

    def plot_pair_wise(self): #Spec 4 plotting the pair wise
        plt.clf()
        plt.scatter(self.data['GrLivArea'], self.data['Price'], label = 'GrLivArea')
        plt.scatter(self.data['BedroomAbvGr'], self.data['Price'], label = 'BedroomAbvGr')
        plt.scatter(self.data['TotalBsmtSF'], self.data['Price'], label = 'TotalBsmtSF')
        plt.scatter(self.data['FullBath'], self.data['Price'], label = 'FullBath')
        plt.title('Features (GrLiveArea, BedroomAbvGr, TotalBsmtSF, FullBath)\n vs. Label (Price)') #adding legend and labels for graph
        plt.ylabel('Price')
        plt.legend(loc = 'upper right')
        plt.show()

    def plot_hist(self): # spec 3 plotting histogram
        plt.clf()
        plt.hist(self.data['Price'])
        plt.title('Price vs. Frequency')
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        plt.show()

    def pred(self): # spec 5
        return self.data.iloc[:, 1:25] @ self.weight # dot product of feature matrix and weight

    def loss(self, predict): #spec 6
        MSE = sum((self.data.iloc[:, 26] - predict)**2) # calculating MSE of predictions
        return MSE/self.size

    def gradient(self, predict): #spec 7
        gradient = 2/self.size * (self.data.iloc[:,1:25].transpose() @ (predict - self.data.iloc[:,26])) # calc gradient
        return gradient
    
    def update(self, gradient): #spec 8
        new_weight = self.weight - self.learning_rate * gradient # calculating new weight using gradient
        return new_weight

    def train_model(self, epochs):
        mse_list = [] # two lists for what im going to plot
        epoch_list = []
        for epoch in range(epochs): # loop for epoch imputs
            epoch_list.append(epoch) 
            y_pred = self.pred() # predict y values
            mse = self.loss(y_pred) # calculate mse
            mse_list.append(mse) # store mse
            grad = self.gradient(y_pred) # calc gradient
            self.weight = self.update(grad) # re-calc weights
            print(f'Epoch #{epoch} MSE: {mse}   {round((epoch/epochs)*100, 5)}%\n') #show cur mse and precentage of training completed
        return epoch_list, mse_list  # return the epochs and mse's to plot

    def get_weight(self):
        return self.weight #returns weight for setting test data to predict

    def set_weight(self, weight):
        self.weight = weight # sets weight 

    def set_learning_rate(self, a):
        self.learning_rate = a # sets learning rate

    def get_data(self):
        return self.data # for plotting against test

if __name__ == '__main__':
    
    modeltest = Model('test.csv') # initialize all necisary models
    model11 = Model('train.csv')
    model12 = Model('train.csv')
    
    weight1 = [0]*24 # set weights and learning coef
    weight2 = [0]*24
    a1 = 10**-11
    a2 = 10**-12

    model11.set_weight(weight1)
    model11.set_learning_rate(a1)

    model12.set_weight(weight2)
    model12.set_learning_rate(a2)

    # Spec 3
    #model11.plot_hist()

    # Spec 4
    #model11.plot_pair_wise()

    # Spec 10
    '''
    model11.set_learning_rate(0.2) # set learning rate to 0.2
    plot = model11.train_model(200000) # train for 200000 epochs
    print(f'Final weight: \n{model11.get_weight}\n') #print final weight
    plt.plot(plot[0], plot[1], label = f'a = {model11.learning_rate}') #plotting
    plt.legend(loc = 'upper right')
    plt.title('Epochs vs. MSE')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.show()
    '''

    # Spec 11/12 
    '''
    plot1 = model11.train_model(400000) # train both models for 400000 epochs
    plot2 = model12.train_model(400000)

    print(f'Final weights: \na1: {model11.weight}\na2: {model12.weight}\n') # get final weights
    plt.plot(plot1[0], plot1[1], label = f'a = {a1}') #plotting
    plt.plot(plot2[0], plot2[1], label = f'a = {a2}')
    plt.title('Iteration vs. MSE')
    plt.xlabel('Iteration')
    plt.ylabel('MSE')
    plt.legend(loc = 'upper right')
    plt.show()
    #'''
    
    #spec 13
    ''' 
    model11.train_model(400000) #train for 400000 epochs
    
    modeltest.set_weight(model11.get_weight()) #set the test model weight to the trained train model
    test_pred = modeltest.pred() # predict values

    plt.scatter(modeltest.get_data()['Id'],modeltest.get_data()['Price'], label = 'Real Price') #plot
    plt.scatter(modeltest.get_data()['Id'], test_pred, label = 'Predicted Prices')
    plt.ylabel('Price')
    plt.xlabel('Id')
    plt.title('Real vs. Predicted Prices')
    plt.legend(loc = 'upper right')


    test_mse = modeltest.loss(test_pred) # get mse

    print(f'Final MSE for test: {test_mse}') # print final MSE

    plt.show()
    '''
