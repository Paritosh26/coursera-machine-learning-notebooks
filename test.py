import pandas as pd
path = 'data/ex1data1.txt'
mydata = pd.read_csv(path)
print(mydata.head())