from babi import *
from ControllerState import ControllerState
from InterfaceWrapper import InterfaceWrapper
import tensorflow as tf
import numpy as np
import random

# Builds the default controller state
def build_default_state():
    state = ControllerState(
        batch_size = tf.shape(inputs)[0],
        class_count = class_count,
        read_vector_count = read_vector_count,
        memory_vector_size = memory_vector_size,
        memory_locations = memory_locations,
        node_counts = node_counts)
    batch_size = tf.shape(inputs)[0]

    return state

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

def vector_norms(x):
  squared_norms = tf.reduce_sum(x * x, axis=2, keep_dims=True)
  return tf.sqrt(squared_norms)

# Calculates a memory access weighting based on a similarity lookup
def content_weighting(memory, key, strength):
    key = tf.reshape(key, [-1, 1, memory_vector_size])
    product = tf.matmul(key, memory, adjoint_b=True)
    memory_norms = vector_norms(memory)
    key_norms = vector_norms(key)
    norm = tf.matmul(key_norms, memory_norms, adjoint_b=True)
    similarity = product / (norm + 0.000001)
    return tf.nn.softmax(tf.reshape(similarity, [-1, memory_locations]))

# Calculates the degree to which each memory location should be retained (Not freed) after reading
def retention_vector(free_gates, read_weightings):
    # Free: ?, read_count
    # weights: ?, read_count, memory_locations
    # result: ?, memory_locations
    free_gates = tf.expand_dims(free_gates, 2)
    return tf.reduce_prod(1 - (free_gates * read_weightings), axis = 1)

# Updates the usage stats for each memory location
def update_usage(old_usage, interface, write_weighting, read_weightings):
    usage = old_usage + write_weighting
    usage -= old_usage * write_weighting
    free_gates = tf.stack([interface.free_gate(x) for x in range(read_vector_count)])

    return usage * retention_vector(free_gates, read_weightings)

# Maps a function to call it on each matrix in a batch
def multi_map_fn(func, args):
    slice_indices = tf.range(tf.shape(args[0])[0])
    return tf.map_fn(lambda x: func(*[arg[x] for arg in args]), slice_indices, dtype = tf.float32)

# Calculates a write weighting based on the degree to which each memory location is "Used up"
def allocation_weighting(usage):
    space = 1 - usage
    most_free, free_indices = tf.nn.top_k(space, k = memory_locations)
    most_free = 1 - most_free
    rolling_space = tf.cumprod(most_free, axis = 1)
    allocation = most_free * rolling_space

    unsorting_indices = tf.map_fn(tf.invert_permutation, free_indices)
    return multi_map_fn(tf.gather, [allocation, unsorting_indices])

# Writes an update to the memory matrix based on the given interface vector
def write_to_memory(interface, memory, usage):
    lookup_key = interface.write_key()
    write_vector = interface.write_vector()
    strength = interface.write_strength()
    mode = tf.expand_dims(interface.allocation_gate(), axis = 1)
    write_gate = interface.write_gate()
    erase_vector = interface.erase_vector()

    lookup_weighting = content_weighting(memory, lookup_key, strength)
    alloc_weighting = allocation_weighting(usage)
    weighting = write_gate * (lookup_weighting * mode + alloc_weighting * (1-mode))
    expanded_weighting = tf.expand_dims(weighting, 2)
    write_vector = tf.expand_dims(write_vector, 1)
    erase_vector = tf.expand_dims(erase_vector, 1)
    memory = memory * (1 - tf.matmul(expanded_weighting, erase_vector)) + tf.matmul(expanded_weighting, write_vector)

    return memory, weighting

# Calculates a new read weighting based on historical links between memory locations
# Locations which are written to in succession will be read in succession
# Also provides a backwards link weighting for a LIFO read order
def link_weightings(links, prev_read, modes):
    prev_read_3d = tf.expand_dims(prev_read, 2)
    forward = tf.reduce_sum(tf.matmul(links, prev_read_3d), axis = 2) * tf.expand_dims(modes[:, 0], 1)
    backward = tf.reduce_sum(tf.matmul(tf.transpose(links, [0, 2, 1]), prev_read_3d), axis = 2) * tf.expand_dims(modes[:, 2], 1)
    return forward, backward

# Generates read vectors for the next step in the controller network
# Read vectors are based on a combination of content lookup and reading in the order things were written
def read_from_memory(interface, memory, links, prev_read):
    read_vectors = []
    weightings = []
    for x in range(read_vector_count):
        lookup_key = interface.read_key(x)
        strength = interface.read_strength(x)
        modes = interface.read_modes(x)
        lookup_weighting = content_weighting(memory, lookup_key, strength) * tf.expand_dims(modes[:, 1], 1)
        forward_weighting, backward_weighting = link_weightings(links, prev_read[:, x, :], modes)
        weighting = forward_weighting + backward_weighting + lookup_weighting
        weightings.append(weighting)
        weighting = tf.tile(weighting, [1, memory_vector_size])
        weighting = tf.reshape(weighting, [tf.shape(weighting)[0], memory_locations, memory_vector_size])
        read_vector = tf.reduce_sum(memory * weighting, axis = 1)

        read_vectors.append(read_vector)
    return tf.stack(read_vectors, axis = 1), tf.stack(weightings, axis = 1)

# Updates the precedence for each location, which indicates how recently each location was written to
def update_precendence_weighting(old_precendence, write_weighting):
    return (1 - tf.reduce_sum(write_weighting, keep_dims = True)) * old_precendence + write_weighting

# Updates the linkage between each of the memory locations
# The linkage indicates how likely a location is to be written to after another location
def update_links(old_links, old_precendence, write_weighting):
    left_weight = tf.expand_dims(write_weighting, 2)
    right_weight = tf.expand_dims(write_weighting, 1)
    right_precedence = tf.expand_dims(old_precendence, 1)
    mod = (1 - left_weight - right_weight)
    return (mod * old_links) + (left_weight * right_precedence)

# Defines a simple feed forward neural network
def basic_network(signal, weights, biases):
    layer = 0
    while layer in weights:
        signal = tf.add(tf.matmul(signal, weights[layer]), biases[layer])
        if layer + 1 in weights:
            signal = tf.nn.relu(signal)

        layer += 1
    return signal

# A generic LSTM network which expects a sequence of vectors
def lstm_network(signal, weights, biases):
    for x in range(len(weights)):
        with tf.variable_scope("LSTM" + str(x)):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(int(weights[x].shape[0]), state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
            value, state = tf.nn.dynamic_rnn(lstm_cell, signal, dtype=tf.float32)

            batch_size = tf.shape(value)[0]
            sequence_weight = tf.expand_dims(weights[x], 0)
            sequence_weight = tf.tile(sequence_weight, [batch_size, 1, 1])

            sequence_bias = biases[x]

            signal = tf.add(tf.matmul(value, sequence_weight), sequence_bias)
            signal = tf.reshape(signal, [batch_size, maximum_sequence_length, int(weights[x].shape[1])])
    return signal

# Calculates a single step in an LSTM network's calculations starting from the given state
def lstm_step(signal, weights, biases, states):
    layer = 0
    while layer in weights:
        with tf.variable_scope("LSTM" + str(layer)):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(int(weights[layer].shape[0]), state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
            signal = tf.expand_dims(signal, 1)
            signal, states[layer] = tf.nn.dynamic_rnn(lstm_cell, signal, initial_state = states[layer], dtype=tf.float32)
            signal = tf.reduce_sum(signal, 1)

            signal = tf.add(tf.matmul(signal, weights[layer]), biases[layer])
            layer += 1
    return signal, states

# Builds a single iteration in the controller network's calculations
# This single iteration takes and produces a tuple of the same dimensions, which is then converted to a ControllerState object
# This allows the step function to be passed into tf.scan
def build_controller_iterator(weights, biases):
    def controller_iteration(prev_state_tuple, signal):
        prev_state = build_default_state()
        prev_state.load_tuple(prev_state_tuple)
        next_state = build_default_state()
        batch_size = tf.shape(prev_state.read_vectors)[0]
        flattened_read_vectors = tf.reshape(prev_state.read_vectors, [batch_size, -1])
        controller_input = tf.concat([signal, flattened_read_vectors], 1)
        controller_input = tf.reshape(controller_input, [batch_size, input_count + (memory_vector_size * read_vector_count)])

        controller_output, next_state.lstm_states = lstm_step(signal, weights, biases, prev_state.lstm_states)
        next_state.output_vector = controller_output[:, : class_count]

        interface_vector = InterfaceWrapper(controller_output[:, class_count :], read_vector_count, memory_vector_size)
        next_state.memory, next_state.write_weighting = write_to_memory(interface_vector, prev_state.memory, prev_state.usage)
        next_state.read_vectors, next_state.read_weightings = read_from_memory(interface_vector, prev_state.memory, prev_state.links, prev_state.read_weightings)
        next_state.precedence = update_precendence_weighting(prev_state.precedence, next_state.write_weighting)
        next_state.links = update_links(prev_state.links, prev_state.precedence, next_state.write_weighting)
        next_state.usage = update_usage(prev_state.usage, interface_vector, next_state.write_weighting, next_state.read_weightings)

        return next_state.to_tuple()
    return controller_iteration

# Defines a network with the additional ability to interface with external memory
def controller_network(input_sequence, weights, biases):
    add_controller_weights(weights)
    controller_iteration = build_controller_iterator(weights, biases)
    input_sequence = tf.transpose(input_sequence, [1, 0, 2])
    result_states = tf.scan(controller_iteration, input_sequence, initializer = build_default_state().to_tuple())
    
    result = build_default_state()
    result.load_tuple(result_states)
    return tf.transpose(result.output_vector, [1, 0, 2])

# Generates a random sequence with an expected output of a delayed echo
def generate_echo_data():
    x = np.eye(class_count)[np.random.choice(class_count, maximum_sequence_length)]
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0

    return (x, y)

# Calculates the average softmax cost of each step in the sequence
# Only considers the prediction and target vectors where there is a target value
def sparse_sequence_softmax_cost(predict, targets):
    cost_mask = tf.sign(tf.reduce_max(tf.abs(targets), reduction_indices=2))
    cost = tf.nn.softmax_cross_entropy_with_logits(logits = predict, labels = targets)
    cost *= cost_mask
    cost = tf.reduce_sum(cost)
    cost /= tf.reduce_sum(cost_mask)
    return cost

# Calculates the ratio between correct predictions in the sequence, and the number of sequences in the batch
def correct_prediction_ratio(predict, targets):
    target_mask = tf.sign(tf.reduce_max(tf.abs(targets), reduction_indices=2))
    predict = tf.argmax(predict, axis = 2)
    targets = tf.argmax(targets, axis = 2)
    matches = tf.to_float(tf.equal(predict, targets))
    return tf.reduce_sum(matches * target_mask) / tf.to_float(tf.shape(predict)[0])

dataset_path = r"datasets\qa1_single-supporting-fact.txt"
babi_io = load_babi_file(dataset_path)

input_count = np.array(babi_io[0][0]).shape[1]
class_count = np.array(babi_io[0][1]).shape[1]
maximum_sequence_length = np.array(babi_io[0][0]).shape[0]
hidden_count = 64

read_vector_count = 4
memory_vector_size = 16
memory_locations = 16
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

learning_rate = .0001
epoch_count = 1000
test_ratio = .1
batch_size = 16

inputs = tf.placeholder("float", [None, maximum_sequence_length, input_count])
targets = tf.placeholder("float", [None, maximum_sequence_length, class_count])

node_counts = [hidden_count, class_count + interface_vector_size]
weights = declare_weights(node_counts)
biases = declare_biases(node_counts)

predict = controller_network(inputs, weights, biases)
cost = sparse_sequence_softmax_cost(predict, targets)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
accuracy = correct_prediction_ratio(predict, targets)

random.shuffle(babi_io)
test_data = babi_io[:int(test_ratio * len(babi_io))]
train_data = babi_io[len(test_data):]
if not batch_size:
    batch_size = len(train_data)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(epoch_count):
        batch_count = int(len(train_data) / batch_size)
        for batch_index in range(batch_count):
            batch = train_data[batch_index * batch_size : (batch_index + 1) * batch_size]
            batch_inputs = np.array([x[0] for x in batch])
            batch_targets = np.array([x[1] for x in batch])

            batch_predict, batch_cost, _ = sess.run([predict, cost, optimizer], feed_dict={inputs: batch_inputs, targets: batch_targets})

        random.shuffle(train_data)

        if epoch % 1 == 0:
            test_inputs = np.array([x[0] for x in test_data])
            test_targets = np.array([x[1] for x in test_data])
            test_cost = sess.run([cost, accuracy], feed_dict={inputs: test_inputs,
                                                  targets: test_targets})
            train_cost = sess.run([cost, accuracy], feed_dict={inputs: np.array([x[0] for x in train_data]),
                                                  targets: np.array([x[1] for x in train_data])})
            print("Epoch " + str(epoch) + " test:     " + str(test_cost))
            print("Epoch " + str(epoch) + " training: " + str(train_cost))