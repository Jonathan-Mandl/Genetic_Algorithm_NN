import numpy as np
import random
import copy
import json

population_size = 100
num_generations = 100
mutation_chance = 0.1

class Model:
    def __init__(self, matrices):
        self.matrices = matrices
        self.b = 0
        self.fitness = 0

    def set_fitness(self, fitness):
        self.fitness = fitness

    def get_fitness(self):
        return self.fitness

    def set_matrices(self, matrices):
        self.matrices = matrices

    def get_matrices(self):
        return self.matrices,self.b
    def set_b(self,b):
        self.b=b

def read_data_file(file_path):
    """
    reads X,y data from file_path
    :param file_path: path to the data file
    :return: X and y data as arrays
    """
    X = []
    y = []
    with open(file_path, "r") as file:
        for line in file.readlines():
            x_string = line[:16]
            x_array = np.array([float(x) for x in x_string]).reshape(16, 1)
            y_label = int(line[19:])
            X.append(x_array)
            y.append(y_label)
    return X, y


def relu(x):
    """
    ReLU activation function
    :param x: input
    :return: max(0, x)
    """
    return np.maximum(0, x)


def sigmoid(z):
    """
    Sigmoid activation function
    :param z: input
    :return: 1 / (1 + exp(-z))
    """
    return 1 / (1 + np.exp(-z))


def split_data(X, y):
    """
    Splits data into 75% train and 25% test sets
    :param X: input data
    :param y: labels
    :return: x_train, y_train, x_test, y_test
    """
    X = np.array(X)  # Convert X to a NumPy array
    y = np.array(y)  # Convert y to a NumPy array
    permutation = np.random.permutation(len(X))
    X = X[permutation]
    y = y[permutation]
    split_index = int(len(X) * 0.75)
    x_train, x_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    return x_train, y_train, x_test, y_test


def xavier_init(shape):
    """
    Xavier weight initialization
    :param shape: shape of the weight matrix
    :return: initialized weight matrix
    """
    n_out,n_in = shape[0], shape[1]
    limit = np.sqrt(6.0 / (n_in + n_out))

    # Initialize the weights from a uniform distribution
    weight_matrix = np.random.uniform(-limit, limit, shape)
    return weight_matrix


def predict(model, X):
    """
    Make predictions using the model
    :param model: model object
    :param X: input data
    :return: array of predictions
    """
    w1,b = model.get_matrices()
    predictions = []
    for x in X:
        # Forward propagation through the model
        z1 = np.dot(w1, x) + b
        output = z1
        # decide prediction according to output of the network
        if output > 0:
            prediction = 1
        else:
            prediction = 0
        predictions.append(prediction)

    return predictions

def calculate_accuracy(predictions, y):
    """
    Calculate accuracy of predictions compared to true labels
    :param predictions: predicted labels
    :param y: true labels
    :return: accuracy value
    """
    correct_predictions = np.sum(predictions == y)
    accuracy = correct_predictions / len(y)
    return accuracy

def evaluate_model(model, x_train, y_train):
    """
    Evaluate the fitness of the model based on its performance on the training data
    :param model: model object
    :param x_train: training data
    :param y_train: training labels
    :return: model with updated fitness value
    """
    predictions = predict(model, x_train)
    model_fitness = calculate_accuracy(predictions, y_train)
    model.set_fitness(model_fitness)
    return model


def fitness(population, x_train, y_train):
    """
    Evaluates the fitness of each model in the population using multiprocessing.

    Args:
        population: List of Model objects.
        x_train: Input data for training.
        y_train: Target values for training.

    Returns:
        sorted_models: List of Model objects sorted by fitness in descending order.
    """
    evaluated_population = []
    for model in population:
        evaluated_model = evaluate_model(model, x_train, y_train)
        evaluated_population.append(evaluated_model)

    sorted_models = sorted(evaluated_population, key=lambda model: model.get_fitness(), reverse=True)
    return sorted_models

def init_population():
    """
    Initialize the population with random weight matrices
    :return: list of model objects
    """
    population = [] # whole population list
    for i in range(population_size):
        # generate new weight matrices for each new model in population
        w1 = xavier_init((1, 16))
        b = np.random.uniform(-1,1)
        matrices = [w1]
        model = Model(matrices) # create new model object based on weight matrices
        model.set_b(b)
        population.append(model) # append model to the population list

    return population

def mutation(model):
    """
    Perform mutation on the weight matrices of the model
    :param model: model object
    """
    d = 20
    w,b= model.get_matrices()
    # iterate over each element of weight matrix
    for cell in range(len(w)):
        # generate random number
        r = np.random.uniform(0, 1)
        # if random number is smaller than mutation chance, generate a mutation
        if r < mutation_chance:
            u = np.random.uniform(0, 1)
            if u < 0.5:
                delta = (2 * u) ** (1.0 / (d + 1)) - 1
            else:
                delta = 1 - (2 * (1 - u)) ** (1.0 / (d + 1))
            w[cell] += delta  # add delta value to element of matrix - this is the mutation
            # ensure cell value remains between -1 and 1
            w[cell] = np.clip(w[cell], -1, 1)

    d = 5
    u = np.random.uniform(0, 1)
    if u < 0.5:
        delta = (2 * u) ** (1.0 / (d + 1)) - 1
    else:
        delta = 1 - (2 * (1 - u)) ** (1.0 / (d + 1))
    b += delta  # add delta value to element of matrix - this is the mutation
    # ensure cell value remains between -1 and 1
    model.set_b(b)


def crossover(model1, model2):
    """
    Perform crossover between two models to create two offspring models
    :param model1: first parent model
    :param model2: second parent model
    :return: two offspring models
    """
    w1,b1= model1.get_matrices()
    w2,b2 = model2.get_matrices()

    num_elements = 16
    # Select random index for crossover
    crossover_idx = np.random.randint(0, num_elements-2)

    # Perform crossover between the two weight matrices
    off1_matrix = w1[:crossover_idx]+w2[crossover_idx:]
    off2_matrix = w2[:crossover_idx]+w1[crossover_idx:]

    offspring1 = Model(off1_matrix)
    offspring2 = Model(off2_matrix)

    offspring1.set_b(b2)
    offspring2.set_b(b1)

    return offspring1, offspring2


def create_next_generation(population,x_train,y_train):
    """
    Create the next generation of the population using selection, crossover, and mutation
    :param population: current population
    :param x_train: training data
    :param y_train: training labels
    :return: new population and best performing network
    """
    population=fitness(population,x_train,y_train)
    best_network=population[0]
    next_generation=[]
    crossover_population=[]
    for i in range(10):
        model=population[i]
        next_generation.append(model)
        for j in range(9): # do 6 times
            model_copy = copy.deepcopy(model)
            crossover_population.append(model_copy)

    for i in range(len(crossover_population)//2):
        parent1=random.choice(crossover_population)
        crossover_population.remove(parent1)
        parent2=random.choice(crossover_population)
        crossover_population.remove(parent2)
        offspring1, offspring2=crossover(parent1,parent2)
        next_generation.append(offspring1)
        next_generation.append(offspring2)

    for model in next_generation[10:]:
        mutation(model)

    return next_generation,best_network

if __name__ == '__main__':
    X, y = read_data_file('nn1.txt')
    x_train, y_train, x_test, y_test=split_data(X,y)
    generation=0
    population=init_population()
    best_network=population[0]
    while generation<num_generations:
        print("Generation: ",generation)
        population,best_network=create_next_generation(population,x_train,y_train)
        current_fitness=best_network.get_fitness()
        print("Best network fitness: ",current_fitness)
        generation += 1
        if current_fitness==1:
            break

    test_predictions=predict(best_network,x_test)
    test_accuracy=calculate_accuracy(test_predictions,y_test)
    print("Best network accuracy on test set: ",test_accuracy)
    w1=best_network.get_matrices()[0]
    b=best_network.get_matrices()[1]
    best_model_data = {
        "matrices": [mat.tolist() for mat in w1],
        "biases": b
    }
    with open("wnet1.txt", "w") as file:
        json.dump(best_model_data, file)
