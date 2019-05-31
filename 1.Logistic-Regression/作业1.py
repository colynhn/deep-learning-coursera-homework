import numpy as np
import matplotlib.pyplot as plt
import h5py
from lr_utils import load_dataset

# 加载数据集：
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()


#将数据集进行reshape,适合输入：
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]

num_px = train_set_x_orig.shape[1]

print(m_train, m_test, num_px)

# 将数据集进行均值化（每个样本的每一列都是像素的列向量集合，故需要除以255）
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.

#以上为作为数据集的预处理。

def initialize_with_zeros(dim):
    
    w = np.zeros((dim, 1))
    b = 0
    return w, b


def sigmoid(z):

    a = 1 / (1 + np.exp(-z))
    return a 

def propagate(w, b, X, Y):

	Z = np.dot(w.T, X) + b
	A = sigmoid(Z)

	m = X.shape[0]
	cost = (- 1 / m) * np.sum(Y * np.log(A) + (1-Y) * (np.log(1-A)))

	dw = (1 / m) * np.dot(X, (A - Y).T)
	db =(1 / m)* np.sum(A - Y)

	grads = {"dw": dw,

			 "db": db}
	return grads, cost


def optimize(w, b, X, Y, num_iterations,learning_rate, print_cost = False):

	costs = []

	for i in range(num_iterations):

		grads, cost = propagate(w, b, X, Y)

		dw = grads["dw"]
		db = grads["db"]

		w = w - learning_rate * dw
		b = b - learning_rate * db

		if i % 100 == 0:

			costs.append(cost)

		if print_cost and i % 100 == 0:

			print("Cost after iteration %i : %f" %(i, cost))

	params = {"w": w, "b": b}

	grads = {"dw": dw, "db": db }

	return params, grads, costs


# 对测试集进行操作
def predict(w, b, X):

	m = X.shape[1]
	Y_predict = np.zeros((1, m))
	w = w.reshape(X.shape[0], 1)

	Z = np.dot(w.T, X) + b
	A = sigmoid(Z) 

	for i in range(X.shape[1]):

		Y_predict[0, i] = 1 if A[0, i] > 0.5 else 0

	return Y_predict 

def model(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate, print_cost):

	w, b = initialize_with_zeros(X_train.shape[0])

	params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations,learning_rate, print_cost)

	w = params["w"]
	b = params["b"]

	Y_predict_test = predict(w, b, X_test)
	Y_predict_train = predict(w, b, X_train) 

	print("训练精度：{}".format(100 - np.mean(np.abs(Y_predict_train - Y_train)) * 100))
	print("测试精度：{}".format(100 - np.mean(np.abs(Y_predict_test - Y_test)) * 100))

	d = {"costs":costs,
		 "Y_predict_train": Y_predict_train,
		 "Y_predict_test": Y_predict_test,
		 "w": w,
		 "b": b,
		 "learning_rate": learning_rate,
		 "num_iterations": num_iterations}


	return d

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.5, print_cost = True)

index = input("请输入0 - 208 的一个数字：")
index = int(index)

plt.imshow(test_set_x[:, index].reshape(num_px, num_px, 3))
change = int(d["Y_predict_test"][0, index])
print ("y = " + str(test_set_y[0, index]) + ", you predicted that it is a \"" 
       + str(classes[change].decode("UTF-8"))   + "\" picture.")

costs = np.squeeze(d["costs"])
plt.plot(costs)
plt.ylabel("cost")
plt.xlabel("iteration(per hundreds)")
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()























 


















