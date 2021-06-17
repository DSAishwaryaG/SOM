#Self Organizing Maps
#columns are attributes and the lines are customers,unsupervised DL model is going 
#to identify some patterns(customers).Basically it does customer segmentation 
#where one of the segments contains the customers that potentially cheated
#One segment is the range of values
#All the customers are input vectors of our NN which are mapped onto the output space
#In between input and output space,there is a neural network
#Each neuron = a vector of weights which is of the same size of vector of 
#customer i.e the vector of 15 elements(CustomerId + 14 other elements)
#For each customer,the output of this customer is the neuron that is close to the customer
#we first pick the closest neuron(most similar neuron of the customer)(winning node),
#then we used a neighbourhood function like Gaussian Neighbourhood Function to update 
#weights of the winning node neighbourhoods to move them closer to the point
#We do this for all the customers and repeat the whole process multiple times,
#this will reduce output space(reduces its dimensions)
#The process ends when the output space stops reducing 
#Finally we obtain a SOM in 2D that contains the identified winning nodes 
#Our target is to find the outliers(fraud) in 2D SOM,because these are far from 
#the majority neurons that follow rules
#Detecting outliers = Mean inter-neuron distance(MID)
#For each neuron - mean of euclidean distance between a neuron and 
#its neighbourhood neurons(we will define neighbourhood manually)
#we will get outliers and finally we do inverse mapping to find which customer 
#originally in the input space are associated with a winning node which is an outlier 
#Class = 0(no) & 1(yes)

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
data = pd.read_csv("Credit_Card_Applications.csv") 

#we are doing this to just distinguish between approved 
#and non-approved customers and not for supervised learning process
#In SOM, we will only use x(find patterns)as we are performing unsupervised learning(no labels)
x = data.iloc[:,:-1].values #all columns except the last one
y = data.iloc[:,-1].values #last column

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
x = sc.fit_transform(x)#fit() = all the info of x like min and max to apply scaling 
 
#Training the SOM - implement SOM from scratch or use a class already developed
from minisom import MiniSom
som = MiniSom(x=10,y=10,input_len=15,sigma=1.0,learning_rate=0.5) 
                #as our observations are few,we will make 10*10 grid(x,y)
                #input_len - 15(CustomerId doesnot help in analysis but to know the fraud)
                #sigma - radius of winning node helps to find MID
som.random_weights_init((x)) #to initialise weights
som.train_random(data=x,num_iteration=100)

#Visualizing the results
#we will see a 2D grid of winning nodes and MID for each neuron
#higher sigma,higher the MID,the more chances of winning node becoming an outlier
#winning node with high MID is the outlier
#larger the MID,colour of winning node is closer to white
from pylab import bone,pcolor,colorbar,plot,show
bone()#empty window to which map is added
#put different winning nodes on the map by adding MIDs of all neurons 
#on the map which SOM identified

pcolor(som.distance_map().T)
#to get things in right order for pcolor,we transpose the matrix(T)
#distance_map() - returns matrix of MIDs(winning nodes)
#adding MIDs and differentiating them through color

colorbar()
markers = ['o','s']
colors = ['r','g']
#unapproved customers - 'o'(red),approved customers - 's'(green)
#association between markers and colors by running a loop through all the customers

for i,v in enumerate(x): 
    #i - index of customers [0-689]
    #v - vector of each customer(row containing all column values of customer)
    w = som.winner(v) #winner() - returns winning node for each customer
    #Plotting the markers on the grid
    plot(w[0]+0.5,# x coordinate at the centre of the square cell of 10*10 grid
         w[1]+0.5,# y coordinate at the centre of the square cell of 10*10 grid
         markers[y[i]],# i=0->y[0] if 0 then markers[0]->'o' elif y[0] is 1 then 
                       #markers[1]->'s'
         markeredgecolor=colors[y[i]],
         markerfacecolor='None',
         markersize=10,
         markeredgewidth=2)
show()
#SOMs are different everytime we run the code.
#data['Class'].value_counts()  
#Finding the frauds   
mappings = som.win_map(x) # to get customers associated with all winning maps
fraud = mappings[(3,7)]
#fraud = np.concatenate((mappings[(1,8)],mappings[(3,2)],mappings[(8,1)]),axis=0)
fraud = sc.inverse_transform(fraud)