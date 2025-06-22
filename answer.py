import os
import math
import random

trainpath = 'training.dat'
testpath = 'testing.dat'


def dataread(path):
    data = []
    with open(path, 'r') as file:
        for line in file:
            data.append([float(x) for x in line.strip().split()])
    return data


traindata = dataread(trainpath)
testdata = dataread(testpath)

# 전달 함수
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# 역전파를 위한 전달 함수의 미분
def derivative_of_sigmoid(x):
    return x * (1 - x)

def entropy_loss(expected, predicted):
    loss = 0
    for i in range(len(expected)):
        loss += expected[i] * math.log(predicted[i])
    return -loss

def one_hot_encode(label, class_count):
    answer = []
    for i in range(class_count):
        if i == label:
            answer.append(1)
        else:
            answer.append(0)
    return answer

# 신경망 클래스 구현
class NeuralNetwork:
    # 생성자를 통한 초기화
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_hidden_weight = []
        for i in range(hidden_nodes):
            row = []
            for j in range(input_nodes):
                row.append(random.uniform(-1, 1))
            self.input_hidden_weight.append(row)

        self.hidden_output_weight = []
        for i in range(output_nodes):
            row = []
            for j in range(hidden_nodes):
                row.append(random.uniform(-1, 1))
            self.hidden_output_weight.append(row)

        self.bias_hidden = []
        for i in range(hidden_nodes):
            self.bias_hidden.append(random.uniform(-1, 1))

        self.bias_output = []
        for i in range(output_nodes):
            self.bias_output.append(random.uniform(-1, 1))

    def feedforward(self, inputs):
        hidden = []
        output = []
        # 은닉층 계산
        for i in range(len(self.input_hidden_weight)):
            sum_i = 0
            for j in range(len(inputs)):
                sum_i += inputs[j] * self.input_hidden_weight[i][j]
            hidden.append(sigmoid(sum_i + self.bias_hidden[i]))

        # 출력층 계산
        for i in range(len(self.hidden_output_weight)):
            sum_h = 0
            for j in range(len(hidden)):
                sum_h += hidden[j] * self.hidden_output_weight[i][j]
            output.append(sigmoid(sum_h + self.bias_output[i]))

        return hidden, output

    def backpropagation(self, inputs, expected_output, learning_rate):
        # 순방향 계산
        hidden, output = self.feedforward(inputs)

        # 출력층 오차 계산
        # 출력층 오차 및 델타 계산
        output_errors = []
        output_deltas = []
        for i in range(len(output)):
            error = expected_output[i] - output[i]
            delta = error * derivative_of_sigmoid(output[i])
            output_errors.append(error)
            output_deltas.append(delta)

        # 은닉층 오차 및 델타 계산
        hidden_errors = []
        hidden_deltas = []
        for j in range(len(hidden)):
            error_sum = 0
            for i in range(len(output)):
                error_sum += output_deltas[i] * self.hidden_output_weight[i][j]
            hidden_errors.append(error_sum)
            delta = error_sum * derivative_of_sigmoid(hidden[j])
            hidden_deltas.append(delta)
        # 가중치 업데이트
        for i in range(len(self.hidden_output_weight)):
            for j in range(len(self.hidden_output_weight[i])):
                self.hidden_output_weight[i][j] += output_deltas[i] * learning_rate * hidden[j]

        for i in range(len(self.input_hidden_weight)):
            for j in range(len(self.input_hidden_weight[i])):
                self.input_hidden_weight[i][j] += hidden_deltas[i] * learning_rate * inputs[j]

        # 편향 업데이트
        for i in range(len(self.bias_output)):
            self.bias_output[i] += learning_rate * output_deltas[i]

        for i in range(len(self.bias_hidden)):
            self.bias_hidden[i] += learning_rate * hidden_deltas[i]

class NeuralNetworkField(NeuralNetwork):
    def train_epochs_time(self, traindata, learning_rate, epochs, labels):
        for epoch in range(epochs):
            loss = 0
            correct = 0
            for i, data_point in enumerate(traindata):
                output = one_hot_encode(labels[i], 3)
                self.backpropagation(data_point, expected_output=output, learning_rate=learning_rate)
                _, cal_output = self.feedforward(data_point)
                loss += entropy_loss(output, cal_output)
                predicted_label = cal_output.index(max(cal_output))
                if labels[i] == predicted_label:
                    correct += 1

            if epoch % 10 == 0:
                average_loss = loss / len(traindata)
                accuracy = correct / len(traindata)
                print(f"{epoch}, Loss: {average_loss}, Accuracy: {accuracy}")

    # ✅ 바깥에 정의된 predict 메서드
    def predict(self, inputs):
        _, output = self.feedforward(inputs)
        return output.index(max(output))
# 신경망 생성 및 훈련
input_nodes = 4
hidden_nodes = 5
output_nodes = 3

nn_modified = NeuralNetworkField(input_nodes, hidden_nodes, output_nodes)

# 훈련 라벨 설정
num_data_per_class = len(traindata) // output_nodes
training_labels = [i // num_data_per_class for i in range(len(traindata))]

# 학습
epochs = 200
learning_rate = 0.01
nn_modified.train_epochs_time(traindata, learning_rate, epochs, training_labels)

# 테스트 라벨 설정
num_data_per_class_test = len(testdata) // output_nodes
test_labels = [i // num_data_per_class_test for i in range(len(testdata))]

# 테스트 정확도 평가
correct = 0
for i, test_point in enumerate(testdata):
    predicted_label = nn_modified.predict(test_point)
    if predicted_label == test_labels[i]:
        correct += 1

accuracy = correct / len(testdata)
print(f"Test Accuracy: {accuracy:.4f}")