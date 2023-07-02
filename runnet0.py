import json
import numpy as np

def relu(x):
    return np.maximum(0, x)

def feed_forward(matrices, biases, X):
    w1, w2, w3, w4 = matrices
    b1, b2, b3, b4 = biases
    predictions = []
    for x in X:
        # Forward propagation through the model
        z1 = np.dot(w1, x) + b1
        h1 = relu(z1)
        z2 = np.dot(w2, h1) + b2
        h2 = relu(z2)
        z3 = np.dot(w3, h2) + b3
        h3 = relu(z3)
        z4 = np.dot(w4, h3) + b4
        output = z4
        # decide prediction according to output of the network
        if output > 0:
            prediction = 1
        else:
            prediction = 0
        predictions.append(prediction)

    return predictions

if __name__ == '__main__':
    with open("testnet0", "r") as file:
        x_test=[]
        for line in file.readlines():
            x_string=line[:16]
            x_array = np.array([float(x) for x in x_string]).reshape(16, 1)
            x_test.append(x_array)

    with open("wnet0.txt", "r") as file:
        best_model_data = json.load(file)

    matrices=best_model_data["matrices"]
    biases= best_model_data["biases"]

    predictions=feed_forward(matrices,biases,x_test)

    with open("predictions0.txt","a") as output_file:
        for pred in predictions:
            output_file.write(str(pred)+"\n")

    print("Classification Done.")
    input("Press Enter to exit...")





