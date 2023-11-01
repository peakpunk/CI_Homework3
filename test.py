import numpy as np
import random

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define a simple feedforward neural network class
# Define a simple feedforward neural network class
class MLPClassifierCustom:
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.weights = []

        # Initialize weights
        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
        for i in range(1, len(layer_sizes)):
            w = np.random.rand(layer_sizes[i-1], layer_sizes[i])
            self.weights.append(w)

    def fit(self, X, y, learning_rate=0.01, epochs=1000):
        for _ in range(epochs):
            for i in range(len(X)):
                input_data = X[i]
                layer_output = [input_data]  # Store the output of each layer

                # Forward pass
                for j in range(len(self.weights)):
                    w = self.weights[j]
                    input_data = sigmoid(np.dot(input_data, w))
                    layer_output.append(input_data)

                # Backpropagation
                error = y[i] - layer_output[-1]
                for j in range(len(self.weights) - 1, -1, -1):
                    output = layer_output[j+1]
                    output_derivative = output * (1 - output)
                    error_term = error * output_derivative
                    self.weights[j] += learning_rate * np.outer(layer_output[j], error_term)
                    error = np.dot(error_term, self.weights[j].T)

    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            input_data = X[i]
            for w in self.weights:
                input_data = sigmoid(np.dot(input_data, w))
            predictions.append(input_data)
        return np.array(predictions)

   

    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            input_data = X[i]
            for w in self.weights:
                input_data = sigmoid(np.dot(input_data, w))
            predictions.append(input_data)
        return np.array(predictions)


# Custom k-fold cross-validation function
def custom_k_fold(X, y, k, clf, scoring='accuracy'):
    n = len(X)
    fold_size = n // k
    scores = []

    for i in range(k):
        start, end = i * fold_size, (i + 1) * fold_size

        X_test, y_test = X[start:end], y[start:end]
        X_train = np.concatenate((X[:start], X[end:]), axis=0)
        y_train = np.concatenate((y[:start], y[end:]), axis=0)

        clf = MLPClassifierCustom(input_size=X_train.shape[1], hidden_layer_sizes=clf.hidden_layer_sizes, output_size=1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        if scoring == 'accuracy':
            accuracy = np.mean(np.round(y_pred) == y_test)
            scores.append(accuracy)

    return np.mean(scores)

# Read data from 'wdbc.txt' and preprocess it
with open('wdbc.txt', 'r') as file:
    lines = file.readlines()

data = [line.strip().split(',') for line in lines]

X = np.array([list(map(float, row[2:])) for row in data])
y = np.array([1 if row[1] == 'M' else 0 for row in data])

# Define the genetic algorithm parameters
population_size = 10
n_generations = 2
mutation_rate = 0.3
crossover_rate = 0.7
n_hidden_layers_range = (1, 5)
hidden_layer_size_range = (5, 100)

# Create individuals
def create_individual():
    n_hidden_layers = random.randint(*n_hidden_layers_range)
    hidden_layer_sizes = [random.randint(*hidden_layer_size_range) for _ in range(n_hidden_layers)]
    return (n_hidden_layers, hidden_layer_sizes)

# Define the fitness function
def evaluate_individual(individual):
    n_hidden_layers, hidden_layer_sizes = individual
    clf = MLPClassifierCustom(input_size=X.shape[1], hidden_layer_sizes=hidden_layer_sizes, output_size=1)
    score = custom_k_fold(X, y, k=10, clf=clf, scoring='accuracy')
    return score

# Initialize the population
population = [create_individual() for _ in range(population_size)]

# Run the Genetic Algorithm
for generation in range(n_generations):
    # Evaluate fitness for each individual
    fitness_scores = [evaluate_individual(ind) for ind in population]

    # Select the top individuals
    n_selected = int(population_size * 0.1)  # Select top 10%
    selected_indices = np.argsort(fitness_scores)[-n_selected:]
    selected_population = [population[i] for i in selected_indices]

    # Create the next generation
    new_population = selected_population.copy()
    while len(new_population) < population_size:
        parent1, parent2 = random.choices(selected_population, k=2)
        crossover_point = random.randint(1, min(len(parent1[1]), len(parent2[1])) - 1)
        child = (parent1[0], parent1[1][:crossover_point] + parent2[1][crossover_point:])
        if random.random() < mutation_rate:
            n_hidden_layers, hidden_layer_sizes = child
            mutated_layer_size = random.randint(*hidden_layer_size_range)
            hidden_layer_sizes[random.randint(0, n_hidden_layers - 1)] = mutated_layer_size
            child = (n_hidden_layers, hidden_layer_sizes)
        new_population.append(child)

    population = new_population

# Find the best individual
best_individual = max(population, key=evaluate_individual)
best_n_hidden_layers, best_hidden_layer_sizes = best_individual

print("Best MLP Structure:")
print("Number of Hidden Layers:", best_n_hidden_layers)
print("Hidden Layer Sizes:", best_hidden_layer_sizes)
print("Best Accuracy:", evaluate_individual(best_individual))