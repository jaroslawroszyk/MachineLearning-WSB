# from sklearn.datasets import load_breast_cancer

# data = load_breast_cancer()

# print(data.DESCR)

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

data = load_iris(as_frame=True)
data.frame['target'] = data.target_names[data.frame['target']]

print(data.DESCR)


plt.plot(data.frame.loc[data.frame.target=='setosa', 'petal length (cm)'], 
         data.frame.loc[data.frame.target=='setosa', 'petal width (cm)'], 
         'ro', label="Iris setosa")
plt.plot(data.frame.loc[data.frame.target=='versicolor', 'petal length (cm)'], 
         data.frame.loc[data.frame.target=='versicolor', 'petal width (cm)'],
         'go', label="Iris versicolor")
plt.plot(data.frame.loc[data.frame.target=='virginica', 'petal length (cm)'], 
         data.frame.loc[data.frame.target=='virginica', 'petal width (cm)'], 
         'bo', label="Iris virginica")
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.grid()
plt.legend()
plt.show()
