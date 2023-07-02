import numpy as np
import random
import copy
import json

population_size = 100
num_generations = 100
mutation_chance = 0.1

class Model:
    def __init__(self, matrices, biases):
        self.matrices = matrices
        self.biases = biases
        self.fitness = 0

    def set_fitness(self, fitness):
        self.fitness = fitness

    def get_fitness(self):
        return self.fitness

    def set_matrices(self, matrices):
        self.matrices = matrices

    def get_biases(self):
        return self.biases

    def set_biases(self, biases):
        self.biases = biases

    def get_matrices(self):
        return self.matrices

def read_data_file(file_path):
    """
    Reads X,y data from the file.

    Args:
        file_path: Path to the data file.

    Returns:
        X: List of input data arrays.
        y: List of target values.
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
    ReLU activation function.

    Args:
        x: Input array.

    Returns:
        Resulting array after applying ReLU activation.
    """
    return np.maximum(0, x)

def split_data(X, y):
    """
    Splits data into train and test sets.

    Args:
        X: List of input data arrays.
        y: List of target values.

    Returns:
        x_train: Input data for training.
        y_train: Target values for training.
        x_test: Input data for testing.
        y_test: Target values for testing.
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
    Xavier weight initialization.

    Args:
        shape: Shape of the weight matrix.

    Returns:
        Initialized weight matrix.
    """
    # Calculate the range of values for weight initialization
    n_out, n_in = shape[0], shape[1]
    limit = np.sqrt(6.0 / (n_in + n_out))

    # Initialize the weights from a uniform distribution
    weight_matrix = np.random.uniform(-limit, limit, shape)
    return weight_matrix

def predict(model, X):
    """
    Performs forward propagation to make predictions using the model.

    Args:
        model: Model object.
        X: Input data array.

    Returns:
        predictions: List of predicted values.
    """
    w1, w2, w3, w4 = model.get_matrices()
    b1, b2, b3, b4 = model.get_biases()
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
        # Decide prediction according to output of the network
        if output > 0:
            prediction = 1
        else:
            prediction = 0
        predictions.append(prediction)

    return predictions

def calculate_accuracy(predictions, y):
    """
    Calculates the accuracy of the predictions.

    Args:
        predictions: List of predicted values.
        y: List of target values.

    Returns:
        accuracy: Accuracy value.
    """
    correct_predictions = np.sum(predictions == y)
    accuracy = correct_predictions / len(y)
    return accuracy

def evaluate_model(model, x_train, y_train):
    """
    Evaluates the model by calculating its fitness using the training data.

    Args:
        model: Model object.
        x_train: Input data for training.
        y_train: Target values for training.

    Returns:
        model: Model object with updated fitness.
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
    evaluated_population=[]
    for model in population:
        evaluated_model=evaluate_model(model, x_train, y_train)
        evaluated_population.append(evaluated_model)

    sorted_models = sorted(evaluated_population, key=lambda model: model.get_fitness(), reverse=True)
    return sorted_models

def init_population():
    """
    Initializes the population with random models.

    Returns:
        population: List of Model objects.
    """
    population = []  # Whole population list
    for i in range(population_size):
        # Generate new weight matrices for each new model in population
        w1 = xavier_init((10, 16))
        b1 = np.random.uniform(1, -1)
        w2 = xavier_init((8,10))
        b2 = np.random.uniform(1, -1)
        w3 = xavier_init((6,8))
        b3 = np.random.uniform(1, -1)
        w4 = xavier_init((1,6))
        b4 = np.random.uniform(1, -1)
        matrices = [w1, w2, w3, w4]
        biases = [b1, b2, b3, b4]
        model = Model(matrices, biases)  # Create new model object based on weight matrices
        population.append(model)  # Append model to the population list

    return population

def mutation(model):
    """
    Applies mutation to the model's weight matrices and biases.

    Args:
        model: Model object.
    """
    d = 25
    for matrix in model.get_matrices():
        # Iterate over each element of the weight matrix
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                # Generate random number
                r = np.random.uniform(0, 1)
                # If random number is smaller than mutation chance, generate a mutation
                if r < mutation_chance:
                    u = np.random.uniform(0, 1)
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (d + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (d + 1))
                    # delta=np.random.normal(0,0.1)
                    matrix[row, col] += delta  # Add delta value to the element of the matrix - this is the mutation
                    # Ensure cell value remains between -1 and 1
                    matrix[row, col] = np.clip(matrix[row, col], -1, 1)
    new_biases = []
    d = 5
    for b in model.get_biases():
        r = np.random.uniform(0, 1)
        # If random number is smaller than mutation chance, generate a mutation
        if r < 0.5:
            u = np.random.uniform(0, 1)
            if u < 0.5:
                delta = (2 * u) ** (1.0 / (d + 1)) - 1
            else:
                delta = 1 - (2 * (1 - u)) ** (1.0 / (d + 1))
            # delta=np.random.normal(0,0.1)
            b += delta  # Add delta value to the element of the matrix - this is the mutation
            # Ensure cell value remains between -1 and 1
        new_biases.append(b)
    model.set_biases(new_biases)

def crossover(model1, model2):
    """
    Performs crossover between two parent models to generate offspring models.

    Args:
        model1: Parent model 1.
        model2: Parent model 2.

    Returns:
        offspring1: Offspring model 1.
        offspring2: Offspring model 2.
    """
    matrices_1 = model1.get_matrices()
    matrices_2 = model2.get_matrices()
    biases_1 = model1.get_biases()
    biases_2 = model2.get_biases()

    off1_matrices = []
    off2_matrices = []

    for w1, w2 in zip(matrices_1, matrices_2):
        num_rows, num_cols = w1.shape

        # Select random index for crossover
        crossover_idx = np.random.randint(0, num_rows * num_cols)

        # Reshape weight matrices to 1D arrays for easier manipulation
        w1_flat = w1.flatten()
        w2_flat = w2.flatten()

        # Perform crossover between the two weight matrices
        off1_matrix_flat = np.concatenate((w1_flat[:crossover_idx], w2_flat[crossover_idx:]))
        off2_matrix_flat = np.concatenate((w2_flat[:crossover_idx], w1_flat[crossover_idx:]))

        # Reshape the offspring matrix back to its original shape
        off1_matrix = off1_matrix_flat.reshape((num_rows, num_cols))
        off2_matrix = off2_matrix_flat.reshape((num_rows, num_cols))

        # Add the offspring matrix to the list of offspring matrices
        off1_matrices.append(off1_matrix)
        off2_matrices.append(off2_matrix)

    b_crossover = random.randint(0, len(biases_1) - 1)
    new_biases1 = biases_1[:b_crossover] + biases_2[b_crossover:]
    new_biases2 = biases_2[:b_crossover] + biases_1[b_crossover:]

    offspring1 = Model(off1_matrices, new_biases1)
    offspring2 = Model(off2_matrices, new_biases2)
    return offspring1, offspring2

def create_next_generation(population, x_train, y_train):
    """
    Creates the next generation of models using selection, crossover, and mutation.

    Args:
        population: List of Model objects.
        x_train: Input data for training.
        y_train: Target values for training.

    Returns:
        next_generation: List of Model objects for the next generation.
        best_model: Best model from the current generation.
    """
    population = fitness(population, x_train, y_train)
    best_model = population[0]
    next_generation = []
    crossover_population = []
    for i in range(10):
        model = population[i]
        next_generation.append(model)
        for j in range(9):  # Do 9 times
            model_copy = copy.deepcopy(model)
            crossover_population.append(model_copy)

    for i in range(len(crossover_population) // 2):
        parent1 = random.choice(crossover_population)
        crossover_population.remove(parent1)
        parent2 = random.choice(crossover_population)
        crossover_population.remove(parent2)
        offspring1, offspring2 = crossover(parent1, parent2)
        next_generation.append(offspring1)
        next_generation.append(offspring2)

    for model in next_generation[10:]:
        mutation(model)

    return next_generation, best_model

if __name__ == '__main__':
    X, y = read_data_file('nn0.txt')
    x_train, y_train, x_test, y_test = split_data(X, y)
    generation = 0
    population = init_population()
    best_model = population[0]
    while generation < num_generations:
        print("Generation: ", generation)
        population, best_model = create_next_generation(population, x_train, y_train)
        current_fitness = best_model.get_fitness()
        print("Best network fitness: ", current_fitness)
        generation += 1

    test_predictions = predict(best_model, x_test)
    test_accuracy = calculate_accuracy(test_predictions, y_test)
    print("Best network accuracy on test set: ", test_accuracy)
    best_model_data = {
        "matrices": [mat.tolist() for mat in best_model.get_matrices()],
        "biases": best_model.get_biases()
    }
    with open("wnet0.txt", "w") as file:
        json.dump(best_model_data, file)
