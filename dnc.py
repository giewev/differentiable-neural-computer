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

def oneplus(x):
    return 1 + tf.log(1 + tf.exp(x))

# A class that wraps a batch of interface vectors in order to provide the specific factors from them
class InterfaceVector(object):
    def __init__(self, vector):
        self.vector = vector

    def read_key(self, index):
        return self.vector[:, index * memory_vector_size : (index + 1) * memory_vector_size]

    def read_strength(self, index):
        start = read_vector_count * memory_vector_size
        return oneplus(self.vector[:, start + index])

    def write_key(self):
        start = read_vector_count * (memory_vector_size + 1)
        return self.vector[:, start : start + memory_vector_size]

    def erase_vector(self):
        start = read_vector_count * memory_vector_size + memory_vector_size + read_vector_count
        return tf.sigmoid(self.vector[:, start : start + memory_vector_size])

    def write_vector(self):
        start = read_vector_count * memory_vector_size + read_vector_count + (memory_vector_size * 2)
        return self.vector[:, start : start + memory_vector_size]

    def free_gate(self, index):
        start = read_vector_count * memory_vector_size + read_vector_count + (memory_vector_size * 3)
        return tf.sigmoid(self.vector[:, start + index])

    def read_modes(self, index):
        start = read_vector_count * memory_vector_size + (read_vector_count * 2) + (memory_vector_size * 3)
        return tf.nn.softmax(self.vector[:, start + ((index - 1) * 3) : start + (index * 3)])

    def allocation_gate(self):
        start = read_vector_count * memory_vector_size + (read_vector_count * 5) + (memory_vector_size * 3)
        return tf.sigmoid(self.vector[:, start])

    def write_gate(self):
        start = read_vector_count * memory_vector_size + (read_vector_count * 5) + (memory_vector_size * 3)
        return tf.sigmoid(self.vector[:, start + 1])

    def write_strength(self):
        start = read_vector_count * memory_vector_size + (read_vector_count * 5) + (memory_vector_size * 3)
        return oneplus(self.vector[:, start + 2])

class ControllerState(object):
    def __init__(self):
        self.read_vectors = tf.zeros([tf.shape(inputs)[0], read_vector_count, memory_vector_size], "float")
        self.memory = tf.zeros([tf.shape(inputs)[0], memory_locations, memory_vector_size], "float")
        self.links = tf.zeros([tf.shape(inputs)[0], memory_locations, memory_locations], "float")
        self.output_vector = tf.zeros([tf.shape(inputs)[0], class_count], "float")
        self.write_weighting = tf.zeros([tf.shape(inputs)[0], memory_locations])
        self.read_weightings = tf.zeros([tf.shape(inputs)[0], read_vector_count, memory_locations])
        self.precedence = tf.zeros([tf.shape(inputs)[0], memory_locations])
        self.usage = tf.zeros([tf.shape(inputs)[0], memory_locations])

    def to_tuple(self):
        return (self.read_vectors,
                self.memory,
                self.links,
                self.output_vector,
                self.precedence,
                self.write_weighting,
                self.read_weightings,
                self.usage)

    def load_tuple(self, tup):
        self.read_vectors = tup[0]
        self.memory = tup[1]
        self.links = tup[2]
        self.output_vector = tup[3]
        self.precedence = tup[4]
        self.write_weighting = tup[5]
        self.read_weightings = tup[6]
        self.usage = tup[7]

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

def retention_vector(free_gates, read_weightings):
    # Free: ?, read_count
    # weights: ?, read_count, memory_locations
    # result: ?, memory_locations
    free_gates = tf.expand_dims(free_gates, 2)
    return tf.reduce_prod(1 - (free_gates * read_weightings), axis = 1)

def update_usage(old_usage, interface, write_weighting, read_weightings):
    usage = old_usage + write_weighting
    usage -= old_usage * write_weighting
    free_gates = tf.stack([interface.free_gate(x) for x in range(read_vector_count)])

    return usage * retention_vector(free_gates, read_weightings)

def multi_map_fn(func, args):
    slice_indices = tf.range(tf.shape(args[0])[0])
    return tf.map_fn(lambda x: func(*[arg[x] for arg in args]), slice_indices)

def allocation_weighting(usage):
    space = 1 - usage
    most_free, free_indices = tf.nn.top_k(space, k = memory_locations)
    most_free = 1 - most_free
    rolling_space = tf.cumprod(most_free, axis = 1)
    allocation = most_free * rolling_space

    # unsorting_indices = batch_apply(tf.invert_permutation, free_indices)
    # return batch_apply(tf.gather, allocation, unsorting_indices)
    unsorting_indices = tf.map_fn(tf.invert_permutation, free_indices)
    return multi_map_fn(tf.gather, [allocation, unsorting_indices])

# Writes an update to the memory matrix based on the given interface vector
def write_to_memory(interface, memory, usage):
    lookup_key = interface.write_key()
    write_vector = interface.write_vector()
    strength = interface.write_strength()
    mode = interface.allocation_gate()
    write_gate = interface.write_gate()
    erase_vector = interface.erase_vector()

    lookup_weighting = content_weighting(memory, lookup_key, strength)
    alloc_weighting = allocation_weighting(usage)
    weighting = tf.transpose(tf.transpose(lookup_weighting, [1, 0]) * mode, [1, 0]) #+ allocation_weighting * (1-mode)
    expanded_weighting = tf.expand_dims(weighting, 2)
    write_vector = tf.expand_dims(write_vector, 1)
    erase_vector = tf.expand_dims(erase_vector, 1)
    # memory = memory * (1 - tf.matmul(weighting, erase_vector)) + tf.matmul(weighting, write_vector)
    memory = memory + tf.matmul(expanded_weighting, write_vector)

    return memory, weighting

def link_weightings(links, prev_read, modes):
    prev_read_3d = tf.expand_dims(prev_read, 2)
    forward = tf.reduce_sum(tf.matmul(links, prev_read_3d), axis = 2) * tf.expand_dims(modes[:, 0], 1)
    backward = tf.reduce_sum(tf.matmul(tf.transpose(links, [0, 2, 1]), prev_read_3d), axis = 2) * tf.expand_dims(modes[:, 2], 1)
    return forward, backward

def read_from_memory(interface, memory, links, prev_read):
    read_vectors = []
    weightings = []
    for x in range(read_vector_count):
        lookup_key = interface.read_key(x)
        strength = interface.read_strength(x)
        modes = interface.read_modes(x)
        lookup_weighting = content_weighting(memory, lookup_key, strength) * tf.expand_dims(modes[:, 1], 1)
        forward_weighting, backward_weighting = link_weightings(links, prev_read[:, x, :], modes)
        weighting = lookup_weighting + forward_weighting + backward_weighting
        weightings.append(weighting)
        weighting = tf.tile(weighting, [1, memory_vector_size])
        weighting = tf.reshape(weighting, [tf.shape(weighting)[0], memory_locations, memory_vector_size])
        read_vector = tf.reduce_sum(memory * weighting, axis = 1)

        read_vectors.append(read_vector)
    return tf.stack(read_vectors, axis = 1), tf.stack(weightings, axis = 1)

def update_precendence_weighting(old_precendence, write_weighting):
    return (1 - tf.reduce_sum(write_weighting)) * old_precendence + write_weighting

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
        next_state.memory, next_state.write_weighting = write_to_memory(interface_vector, prev_state.memory, prev_state.usage)
        next_state.read_vectors,next_state.read_weightings = read_from_memory(interface_vector, prev_state.memory, prev_state.links, prev_state.read_weightings)
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
    result_states = tf.scan(controller_iteration, input_sequence, initializer = ControllerState().to_tuple())

    return result_states[3]

# Generates a random sequence with an expected output of a delayed echo
def generate_echo_data():
    x = np.eye(class_count)[np.random.choice(class_count, maximum_sequence_length)]
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0

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
hidden_count = 64
class_count = 82

echo_step = 5

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

learning_rate = .1
epoch_count = 1000
test_ratio = .2
batch_size = 16

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

            max_predict, batch_predict, batch_cost, _ = sess.run([tf.reduce_max(predict), predict, cost, optimizer], feed_dict={inputs: batch_inputs, targets: batch_targets})
            print(max_predict)

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