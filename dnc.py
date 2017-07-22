import tensorflow as tf
import numpy as np
import random
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

def load_babi_file(path):
    with open(path) as f:
        lines = f.readlines()
        lines = [x.split(' ') for x in lines]
        stories = []
        for x in lines:
            if int(x[0]) == 1:
                stories.append([])
            stories[-1].append(x)
        return stories

def is_numeric(string):
    try:
        int(string)
        return True
    except:
        return False

def build_vectorization(stories):
    ids = dict()
    next_id = 0
    for x in stories:
        for y in x:
            if y not in ids:
                ids[y] = next_id
                next_id = next_id + 1
    ids["_"] = next_id
    return ids

def one_hot(size, index):
    a = np.zeros(shape = (size,), dtype = float)
    a[index] = 1
    return a

def vectorize_babi_file(stories):
    stories = [[z for y in x for z in y if not is_numeric(z)] for x in stories]
    for x in stories:
        y = 0
        while y < len(x):
            if "?" in x[y]:
                x[y] = x[y].replace("?", "")
                x.insert(y + 1, "?")
                y += 1
            if "." in x[y]:
                x[y] = x[y].replace(".", "")
                x.insert(y + 1, ".")
                y += 1
            y += 1
    ids = build_vectorization(stories)
    stories = [[one_hot(len(ids), ids[y]) for y in x] for x in stories]
    return (ids, stories)

def build_babi_targets(stories, ids):
    targeted_stories = []
    for story in stories:
        targets = []
        for ind, word in enumerate(story):
            target = np.zeros(shape = (len(ids),), dtype = float)
            if ind > 0 and np.all(np.equal(story[ind - 1], one_hot(len(ids), ids["?"]))):
                target = word
                story[ind] = one_hot(len(ids), ids["_"])
            targets.append(target)
        targeted_stories.append(targets)
    return (stories, targeted_stories)

def zero_pad_sequence(sequence, length):
    while len(sequence) < length:
        sequence.append(np.zeros(shape = np.shape(sequence[0]), dtype = float))
    return sequence

# A class that wraps a batch of interface vectors in order to provide the specific factors from them
class InterfaceVector(object):
    def __init__(self, vector):
        self.vector = vector

    def read_key(self, index):
        return self.vector[:, index * memory_vector_size : (index + 1) * memory_vector_size]

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

class ControllerState(object):
    def __init__(self):
        self.read_vectors = tf.zeros([tf.shape(inputs)[0], read_vector_count, memory_vector_size], "float")
        self.memory = tf.zeros([tf.shape(inputs)[0], memory_locations, memory_vector_size], "float")
        self.links = tf.zeros([tf.shape(inputs)[0], memory_locations, memory_locations], "float")
        self.output_vector = tf.zeros([tf.shape(inputs)[0], class_count], "float")

    def to_tuple(self):
        return (self.read_vectors, self.memory, self.links, self.output_vector)

    def load_tuple(self, tup):
        self.read_vectors = tup[0]
        self.memory = tup[1]
        self.links = tup[2]
        self.output_vector = tup[3]

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
    similarity = product / norm
    return tf.reshape(similarity, [-1, memory_locations])

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
    weighting = tf.transpose(lookup_weighting, [1, 0]) * mode #+ allocation_weighting * (1-mode)
    memory = memory * (1 - tf.matmul(weighting, erase_vector)) + tf.matmul(weighting, write_vector)

    return memory

def read_from_memory(interface, memory):
    read_vectors = []
    for x in range(read_vector_count):
        lookup_key = interface.read_key(x)
        strength = interface.read_strength(x)
        weighting = content_weighting(memory, lookup_key, strength)
        weighting = tf.tile(weighting, [1, memory_vector_size])
        weighting = tf.reshape(weighting, [tf.shape(weighting)[0], memory_locations, memory_vector_size])
        read_vector = tf.reduce_sum(memory * weighting, axis = 1)

        read_vectors.append(read_vector)
    return tf.stack(read_vectors, axis = 1)

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
    for x in range(len(weights)):
        with tf.variable_scope("LSTM" + str(x)):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(int(weights[x].shape[0]), state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
            value, state = tf.nn.dynamic_rnn(lstm_cell, signal, dtype=tf.float32)

            batch_size = tf.shape(value)[0]
            sequence_weight = tf.tile(weights[x], [batch_size, 1])
            sequence_weight = tf.reshape(sequence_weight, [batch_size, tf.shape(weights[x])[0], tf.shape(weights[x])[1]])

            sequence_bias = tf.tile(biases[x], [batch_size * maximum_sequence_length])
            sequence_bias = tf.reshape(sequence_bias, [batch_size, maximum_sequence_length, tf.shape(biases[x])[0]])

            signal = tf.add(tf.matmul(value, sequence_weight), sequence_bias)
            signal = tf.reshape(signal, [batch_size, maximum_sequence_length, int(weights[x].shape[1])])
    return signal

def build_controller_iterator(weights, biases):
    def controller_iteration(prev_state_tuple, signal):
        prev_state = ControllerState()
        prev_state.load_tuple(prev_state_tuple)
        next_state = ControllerState()
        flattened_read_vectors = tf.reshape(prev_state.read_vectors, [tf.shape(prev_state.read_vectors)[0], -1])
        controller_input = tf.concat([signal, flattened_read_vectors], 1)
        controller_output = basic_network(controller_input, weights, biases)
        next_state.output_vector = tf.matmul(controller_output[:, : class_count], weights['out'])
        
        interface_vector = InterfaceVector(tf.matmul(controller_output[:, class_count :], weights['interface']))
        next_state.memory = write_to_memory(interface_vector, prev_state.memory)
        next_state.read_vectors = read_from_memory(interface_vector, prev_state.memory)

        return next_state.to_tuple()
    return controller_iteration

# Defines a network with the additional ability to interface with external memory
def controller_network(input_sequence, weights, biases):
    add_controller_weights(weights)
    controller_iteration = build_controller_iterator(weights, biases)
    input_sequence = tf.transpose(input_sequence, [1, 0, 2])
    result_states = tf.scan(controller_iteration, input_sequence, initializer = ControllerState().to_tuple())

    result_outputs = result_states[3]
    return result_outputs

# Generates a random sequence with an expected output of a delayed echo
def generate_echo_data():
    x = np.array(np.random.choice(2, maximum_sequence_length, p=[0.5, 0.5]))
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0

    x = np.reshape(x, [-1, 1])
    y = np.reshape(y, [-1, 1])
    return (x, y)

def sparse_sequence_softmax_cost(predict, targets):
    cost_mask = tf.sign(tf.reduce_max(tf.abs(targets), reduction_indices=2))
    cost_mask = tf.transpose(cost_mask, [1, 0])
    cost = tf.nn.softmax_cross_entropy_with_logits(logits = predict, labels = targets)
    cost *= cost_mask
    cost = tf.reduce_sum(cost)
    cost /= tf.reduce_sum(cost_mask)
    return cost

def correct_prediction_ratio(predict, targets):
    target_mask = tf.sign(tf.reduce_max(tf.abs(targets), reduction_indices=2))
    predict = tf.argmax(predict, axis = 2)
    targets = tf.argmax(targets, axis = 2)
    predict = tf.transpose(predict, [1, 0])
    matches = tf.to_float(tf.equal(predict, targets))
    return tf.reduce_sum(matches * target_mask) / tf.to_float(tf.shape(predict)[0])

babi_data = load_babi_file(r"C:\Users\Ian\Downloads\babi_tasks_1-20_v1-2.tar-20170708T211118Z-001\babi_tasks_1-20_v1-2.tar\tasks_1-20_v1-2\en-10k\qa1_single-supporting-fact_train.txt")
babi_ids, babi_data = vectorize_babi_file(babi_data)
babi_data, babi_targets = build_babi_targets(babi_data, babi_ids)
sequence_lengths = [len(x) for x in babi_data]
maximum_sequence_length = max(sequence_lengths)
babi_data = [zero_pad_sequence(x, maximum_sequence_length) for x in babi_data]
babi_targets = [zero_pad_sequence(x, maximum_sequence_length) for x in babi_targets]
babi_io = list(zip(babi_data, babi_targets))

input_count = 82
hidden_count = 100
class_count = 82

echo_step = 5

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

learning_rate = .01
epoch_count = 1000
test_ratio = .2
batch_size = 30

inputs = tf.placeholder("float", [None, maximum_sequence_length, input_count])
targets = tf.placeholder("float", [None, maximum_sequence_length, class_count])

node_counts = [input_count + (read_vector_count * memory_vector_size), hidden_count, class_count + interface_vector_size]
lstm_dimensions = [hidden_count, hidden_count, class_count]
weights = declare_weights(node_counts)
biases = declare_biases(node_counts)

predict = controller_network(inputs, weights, biases)
# cost = tf.reduce_mean(tf.squared_difference(predict, targets))
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

            _ = sess.run([optimizer], feed_dict={inputs: batch_inputs, targets: batch_targets})

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