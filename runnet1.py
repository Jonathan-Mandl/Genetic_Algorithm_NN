import json
import numpy as np

def feed_forward(matrices, biases, X):
    w = matrices[0]
    b= biases
    predictions = []
    for x in X:
        # Forward propagation through the model
        z = np.dot(w, x) + b
        output=z
        # decide prediction according to output of the network
        if output > 0:
            prediction = 1
        else:
            prediction = 0
        predictions.append(prediction)

    return predictions

if __name__ == '__main__':
    with open("testnet1", "r") as file:
        x_test=[]
        for line in file.readlines():
            x_string=line[:16]
            x_array = np.array([float(x) for x in x_string]).reshape(16, 1)
            x_test.append(x_array)

    with open("wnet1.txt", "r") as file:
        best_model_data = json.load(file)

    matrices=best_model_data["matrices"]
    biases= best_model_data["biases"]

    predictions=feed_forward(matrices,biases,x_test)

    with open("predictions1.txt","a") as output_file:
        for pred in predictions:
            output_file.write(str(pred)+"\n")

    print("Classification Done.")
    input("Press Enter to exit...")



