import numpy as np
import random
import matplotlib.pyplot as plt
import os

# Define data
# หาตำแหน่งของไฟล์ที่รันโค้ด
scriptpath = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(scriptpath, 'wdbc.txt')

# อ่านไฟล์ 'wdbc.txt'
with open(filename, 'r') as testFile:
    lines = testFile.readlines()

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

class MLPClassifier:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        # Define the network architecture and learning rate
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights for input to hidden and hidden to output layers
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)

    def feed_forward(self, X):
        # Compute the output of the hidden layer
        self.hidden_input = np.dot(X, self.weights_input_hidden)
        self.hidden_output = sigmoid(self.hidden_input)

        # Compute the output of the output layer
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output)
        self.output = sigmoid(self.output_input)

        return self.output

    def backpropagation(self, X, y):
        # Compute the error in the output layer
        self.error = y - self.output
        self.delta_output = self.error * sigmoid_derivative(self.output)

        # Compute the error in the hidden layer
        self.error_hidden = self.delta_output.dot(self.weights_hidden_output.T)
        self.delta_hidden = self.error_hidden * sigmoid_derivative(self.hidden_output)

        # Update weights for the hidden to output and input to hidden layers
        self.weights_hidden_output += self.hidden_output.T.dot(self.delta_output) * self.learning_rate
        self.weights_input_hidden += X.T.dot(self.delta_hidden) * self.learning_rate

    def train(self, X, y, epochs):
        for _ in range(epochs):
            for i in range(len(X)):
                # Feed Forward
                output = self.feed_forward(X[i])
                # Backpropagation
                self.backpropagation(X[i], y[i])

    def predict(self, X):
        return np.round(self.feed_forward(X))

# Load your data and create X, y as you did in your original code.

# Define k-fold cross-validation
def k_fold_cross_validation(X, y, model, k):
    data_size = len(X)
    fold_size = data_size // k
    accuracies = []

    for i in range(k):
        # Split data into training and testing sets
        test_start = i * fold_size
        test_end = (i + 1) * fold_size
        test_data_X = X[test_start:test_end]
        test_data_y = y[test_start:test_end]

        train_data_X = np.concatenate([X[:test_start], X[test_end:]])
        train_data_y = np.concatenate([y[:test_start], y[test_end:]])

        model.train(train_data_X, train_data_y, epochs=100)  # Adjust the number of epochs as needed

        correct = 0
        total = len(test_data_y)
        for j in range(total):
            prediction = model.predict(test_data_X[j])
            if prediction == test_data_y[j]:
                correct += 1

        accuracy = correct / total
        accuracies.append(accuracy)

    return accuracies

# Hyperparameter optimization using a genetic algorithm
population_size = 100
generations = 100
mutation_rate = 0.1
crossover_rate = 0.9
hiddenlayer_range = (1, 8)
hiddenlayer_size_range = (8, 100)

# Create and initialize the population
def initialize_population(input_dim, output_dim):
    population = []
    for _ in range(population_size):
        num_hidden_layers = random.randint(hiddenlayer_range[0], hiddenlayer_range[1])
        hidden_layer_sizes = [random.randint(hiddenlayer_size_range[0], hiddenlayer_size_range[1]) for _ in range(num_hidden_layers)]
        mlp = create_mlp(input_dim, hidden_layer_sizes, output_dim)
        population.append(mlp)
    return population

# Create an MLP with the specified architecture
def create_mlp(input_dim, hidden_layer_sizes, output_dim):
    mlp = MLPClassifier(input_size=input_dim, hidden_size=hidden_layer_sizes[0], output_size=output_dim, learning_rate=0.1)

    if len(hidden_layer_sizes) > 1:
        for i in range(1, len(hidden_layer_sizes)):
            next_hidden_size = hidden_layer_sizes[i]
            weights = np.random.rand(mlp.hidden_size, next_hidden_size)
            mlp.weights_hidden_output = weights
            mlp.hidden_size = next_hidden_size

    return mlp

# Select parent MLPs for crossover
def select_parents(population, fitness_scores):
    probabilities = [fitness / sum(fitness_scores) for fitness in fitness_scores]
    parents = []
    for _ in range(len(population) // 2):
        parent1 = random.choices(population, probabilities)[0]
        parent2 = random.choices(population, probabilities)[0]
        parents.append((parent1, parent2))
    return parents

# Mutate an MLP
def mutate(mlp, mutation_rate):
    for layer in mlp.weights_input_hidden, mlp.weights_hidden_output:
        mask = np.random.rand(*layer.shape) < mutation_rate
        layer[mask] += np.random.uniform(-0.5, 0.5, size=layer.shape)[mask]
    return mlp

# Select the best MLP from the population
def select_best_mlp(population, fitness_scores):
    best_index = fitness_scores.index(max(fitness_scores))
    best_mlp = population[best_index]
    return best_mlp

def calculate_fitness(mlp, X, y):
    accuracies = k_fold_cross_validation(X, y, mlp, k=5)  # You can adjust the value of k as needed.
    average_accuracy = np.mean(accuracies)
    return average_accuracy

def crossover(parents):
    parent1, parent2 = parents
    child = create_mlp(input_size, parent1.hidden_size, output_size)
    
    for layer in child.weights_input_hidden, child.weights_hidden_output:
        for i in range(layer.shape[0]):
            for j in range(layer.shape[1]):
                if random.random() < 0.5:
                    layer[i, j] = parent1.weights_input_hidden[i, j]
                else:
                    layer[i, j] = parent2.weights_input_hidden[i, j]
    
    return child

num_samples = 100
num_features = 10

data = [line.strip().split(',') for line in lines]

X = np.array([list(map(float, row[2:])) for row in data])
y = np.array([1 if row[1] == 'M' else 0 for row in data])

# Initialize the population
input_size = X.shape[1]
output_size = 1  # Assuming binary classification
population = initialize_population(input_size, output_size)
best_fitness_is_satisfactory = False  # Set to True if a satisfactory fitness is reached

for generation in range(generations):
    fitness_scores = [calculate_fitness(mlp, X, y) for mlp in population]
    best_mlp = select_best_mlp(population, fitness_scores)
    print(f"Generation {generation}: Best Fitness - {max(fitness_scores)}")

    if best_fitness_is_satisfactory:
        break

    parents = select_parents(population, fitness_scores)
    new_population = []
    while len(new_population) < population_size:
        parent1, parent2 = random.choice(parents), random.choice(parents)
        child = crossover(parent1, parent2)
        child = mutate(child, mutation_rate)
        new_population.append(child)

    population = new_population

# Initialize lists to store generation and best fitness values
generation_numbers = []
best_fitness_values = []

# Inside the loop for generations, add the following code to record the best fitness and generation number:
best_fitness = max(fitness_scores)
best_mlp = select_best_mlp(population, fitness_scores)
generation_numbers.append(generation)
best_fitness_values.append(best_fitness)

# After the loop, you can plot the best fitness values:
plt.figure()
plt.plot(generation_numbers, best_fitness_values, marker='o')
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.title('Best Fitness over Generations')
plt.grid(True)
plt.show()




# The best MLP can be found in the variable 'best_mlp'.
# You can access its architecture and weights to get the desired information.
# For example:
print("Best MLP Architecture - Hidden Layers:", best_mlp.hidden_size)
print("Best MLP Architecture - Hidden Layer Sizes:", best_mlp.weights_input_hidden.shape[1])


