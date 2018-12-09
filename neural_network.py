import numpy as np
import scipy.special
import matplotlib.pyplot as plt

train_file_path = 'D:\\Study\\ML&CV\\TensorFlow\\MNIST\\mnist_train.csv'
test_file_path = 'D:\\Study\\ML&CV\\TensorFlow\\MNIST\\mnist_test.csv'

class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.onodes = outputnodes
        self.hnodes = hiddennodes

        #使用正态分布采样权重，均值为0.0，表准方差为传入链接数目的开方
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes) )
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes) )

        self.lr = learningrate

        self.activation_func = lambda x:scipy.special.expit(x)
        pass

    def train(self, input_list, targets_list):
        #将一维的输入向量转化为二维数组
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        #计算隐藏层的组合输入
        hidden_inputs = np.dot(self.wih, inputs)
        #计算隐藏层的输出
        hidden_outputs = self.activation_func(hidden_inputs)
        #计算输出层的组合输入
        final_inputs = np.dot(self.who, hidden_outputs)
        #计算输出层的最终输出
        final_outputs = self.activation_func(final_inputs)

        #计算输出层的误差
        output_errors = targets - final_outputs
        #计算隐藏层的误差
        hidden_errors = np.dot(self.who.T, output_errors)

        #更新隐藏层到输出层间的权重
        self.who += self.lr * np.dot((output_errors * final_outputs * (1 - final_outputs )),
                                     np.transpose(hidden_outputs) )
        #更新输入层到隐藏层的权重
        self.wih += self.lr * np.dot( (hidden_errors * hidden_outputs * (1 - hidden_outputs )),
                                    np.transpose(inputs) )



    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_func(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_func(final_inputs)

        return final_outputs



input_nodes = 784
hidden_nodes = 100
output_nodes = 10

learning_rate = 0.3

nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

#load MNIST traing data
training_data_file = open(train_file_path, "r")
training_data_list = training_data_file.readlines()
training_data_file.close()


epochs = 5

for e in range(epochs):
    for recoder in training_data_list:
        all_values = recoder.split(',')
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99 ) + 0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        nn.train(inputs, targets)
        pass
    pass

#load MNIST test data
test_data_file = open(test_file_path, "r")
test_data_list = test_data_file.readlines()
test_data_file.close()

scorecard = []

for recoder in test_data_list:
    all_values = recoder.split(',')
    correct_label = int(all_values[0])
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99 ) + 0.01
    outputs = nn.query(inputs)
    label = np.argmax(outputs)
    if ( label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass

scorecard_array = np.asarray(scorecard)
print("performance = ", scorecard_array.sum() / scorecard_array.size)



