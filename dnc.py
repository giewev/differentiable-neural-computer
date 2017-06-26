import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

input_count = 1
hidden_count = 5
class_count = 1

read_vector_count = 1
memory_vector_size = 10
memory_locations = 10
interface_vector_dimensions = [
    read_vector_count * memory_vector_size, # Read keys
    read_vector_count, # Read strengths
    memory_vector_size, # Write key
    memory_vector_size, # Erase vector
    memory_vector_size, # Write vector
    read_vector_count, # Free gates
    read_vector_count * 3, # read modes
    1, # Allocation gate
    1, # Write gate
    1] # Write strength

interface_vector_size = sum(interface_vector_dimensions)

maximum_sequence_length = 28
echo_step = 5

learning_rate = 0.01
epoch_count = 10
batch_size = 100

inputs = tf.placeholder("float", [None, maximum_sequence_length, input_count])
targets = tf.placeholder("float", [None, maximum_sequence_length, class_count])

# A class that wraps a batch of interface vectors in order to provide the specific factors from them
class InterfaceVector(object):
    def __init__(self, vector):
        self.vector = vector

    def read_key(self, index):
        return self.vector[:, (index - 1) * memory_vector_size : index * memory_vector_size]

    def read_strength(self, index):
        start = read_vector_count * memory_vector_size
        return self.vector[:, start + index]

    def write_key(self):
        start = read_vector_count * (memory_vector_size + 1)
        return self.vector[:, start : start + memory_vector_size]

    def erase_vector(self):
        start = read_vector_count * memory_vector_size + memory_vector_size + read_vector_count
        return self.vector[:, start : start + memory_vector_size]

    def write_vector(self):
        start = read_vector_count * memory_vector_size + read_vector_count + (memory_vector_size * 2)
        return self.vector[:, start : start + memory_vector_size]

    def free_gate(self, index):
        start = read_vector_count * memory_vector_size + read_vector_count + (memory_vector_size * 3)
        return self.vector[:, start + index]

    def read_modes(self, index):
        start = read_vector_count * memory_vector_size + (read_vector_count * 2) + (memory_vector_size * 3)
        return self.vector[:, start + ((index - 1) * 3) : start + (index * 3)]

    def allocation_gate(self):
        start = read_vector_count * memory_vector_size + (read_vector_count * 5) + (memory_vector_size * 3)
        return self.vector[:, start]

    def write_gate(self):
        start = read_vector_count * memory_vector_size + (read_vector_count * 5) + (memory_vector_size * 3)
        return self.vector[:, start + 1]

    def write_strength(self):
        start = read_vector_count * memory_vector_size + (read_vector_count * 5) + (memory_vector_size * 3)
        return self.vector[:, start + 2]

# Creates a set of weights based on the given layer sizes
def declare_weights(node_counts):
    weights = dict()
    for x in range(len(node_counts) - 1):
        weights[x] = tf.Variable(tf.random_normal([node_counts[x], node_counts[x + 1]]))
    return weights

# Creates a set of biases based on the given layer sizes
def declare_biases(node_counts):
    biases = dict()
    for x in range(len(node_counts) - 1):
        biases[x] = tf.Variable(tf.random_normal([node_counts[x + 1]]))
    return biases

# Add special weights which are applied to the final controller output
# These weights will produce the output and interface vectors respectively
def add_controller_weights(weights):
    weights['out'] = tf.Variable(tf.random_normal([class_count, class_count]))
    weights['interface'] = tf.Variable(tf.random_normal([interface_vector_size, interface_vector_size]))

# Calculates the cosine similarity between two vectors
def cosine_similarity(a, b):
    norm_a = tf.nn.l2_normalize(a,0)        
    norm_b = tf.nn.l2_normalize(b,0)
    return tf.reduce_sum(tf.multiply(norm_a,norm_b))

# Calculates the cosine similarity between each row of a matrix and a given vector
def row_cosine_similarity(x):
    memory = x[0]
    key = x[1]
    strength = x[2]

    def check_similarity(row):
        return cosine_similarity(row, key)
    return strength * tf.map_fn(check_similarity, memory)

# Calculates a memory access weighting based on a similarity lookup
def content_weighting(memory, key, strength):
    return tf.map_fn(row_cosine_similarity, (memory, key, strength))

# Writes an update to the memory matrix based on the given interface vector
def write_to_memory(interface, memory):
    lookup_key = interface.write_key()
    write_vector = interface.write_vector()
    strength = interface.write_strength()
    mode = interface.allocation_gate()
    write_gate = interface.write_gate()
    erase_vector = interface.erase_vector()

    lookup_weighting = content_weighting(memory, lookup_key, strength)
    allocation_weighting = None # Replace with available space heuristic
    weighting = lookup_weighting * mode + allocation_weighting * (1-mode)
    memory = memory * (1 - tf.matmul(weighting, erase_vector)) + tf.matmul(weighting, write_vector)

    return memory

# Defines a simple feed forward neural network
def basic_network(signal, weights, biases):
    layer = 0
    while layer in weights:
        signal = tf.add(tf.matmul(signal, weights[layer]), biases[layer])
        if layer + 1 in weights:
            signal = tf.nn.relu(signal)
        layer += 1
    return signal

def lstm_network(signal, weights, biases):
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_count, state_is_tuple=True)
    value, state = tf.nn.dynamic_rnn(lstm_cell, signal, dtype=tf.float32)

    batch_size = tf.shape(value)[0]
    sequence_weight = tf.tile(weights[0], [batch_size, 1])
    sequence_weight = tf.reshape(sequence_weight, [batch_size, tf.shape(weights[0])[0], tf.shape(weights[0])[1]])
    # sequence_weight = tf.transpose(sequence_weight, [1, 0, 2])

    sequence_bias = tf.tile(biases[0], [batch_size * maximum_sequence_length])
    sequence_bias = tf.reshape(sequence_bias, [batch_size, maximum_sequence_length, tf.shape(biases[0])[0]])

    return tf.add(tf.matmul(value, sequence_weight), sequence_bias)

# Defines a network with the additional ability to interface with external memory
def controller_network(input_sequence, weights, biases):
    add_controller_weights(weights)
    outputs = []
    read_vectors = tf.zeros([tf.shape(inputs)[0], read_vector_count, memory_vector_size], "float")
    memory = tf.zeros([tf.shape(inputs)[0], memory_locations, memory_vector_size], "float")
    links = tf.zeros([tf.shape(inputs)[0], memory_locations, memory_locations], "float")

    for x in range(maximum_sequence_length):
        flattened_read_vectors = tf.reshape(read_vectors, [tf.shape(read_vectors)[0], -1])
        current_input_vector = tf.reshape(input_sequence[:, x, :], [tf.shape(input_sequence)[0], tf.shape(input_sequence)[2]])
        controller_input = tf.concat([current_input_vector, flattened_read_vectors], 1)
        controller_output = basic_network(controller_input, weights, biases)
        outputs.append(tf.matmul(controller_output[:, : class_count], weights['out']))
        interface_vector = InterfaceVector(tf.matmul(controller_output[:, class_count :], weights['interface']))

        memory = write_to_memory(interface_vector, memory)

    outputs = tf.reshape(outputs, [-1, maximum_sequence_length, class_count])
    return outputs

# Generates a random sequence with an expected output of a delayed echo
def generate_echo_data():
    x = np.array(np.random.choice(2, maximum_sequence_length, p=[0.5, 0.5]))
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0

    x = np.reshape(x, [-1, 1])
    y = np.reshape(y, [-1, 1])
    return (x, y)



node_counts = [input_count + (read_vector_count * memory_vector_size), hidden_count, class_count + interface_vector_size]
lstm_dimensions = [hidden_count, class_count]
weights = declare_weights(lstm_dimensions)
biases = declare_biases(lstm_dimensions)

predict = lstm_network(inputs, weights, biases)
cost = tf.reduce_mean(tf.squared_difference(predict, targets))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epoch_count):
        avg_cost = 0.
        batch_count = int(mnist.train.num_examples / batch_size)
        for x in range(batch_count):
            batch = [generate_echo_data() for x in range(batch_size)]
            batch_inputs = np.array([x[0] for x in batch])
            batch_targets = np.array([x[1] for x in batch])
            # batch_inputs, batch_targets = mnist.train.next_batch(batch_size)
            # batch_inputs, batch_targets = convert_batch_to_sequence(batch_inputs, batch_targets, batch_size)

            _, c = sess.run([optimizer, cost], feed_dict={inputs: batch_inputs,
                                                          targets: batch_targets})
            avg_cost += c / batch_count
        print(avg_cost)