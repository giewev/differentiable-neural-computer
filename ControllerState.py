import tensorflow as tf

class ControllerState(object):
    def __init__(self,
        batch_size = 1,
        class_count = 1,
        read_vector_count = 1,
        memory_vector_size = 1,
        memory_locations = 1,
        node_counts = [1, 1]):

        self.read_vectors = tf.zeros([batch_size, read_vector_count, memory_vector_size], "float")
        self.memory = tf.zeros([batch_size, memory_locations, memory_vector_size], "float")
        self.links = tf.zeros([batch_size, memory_locations, memory_locations], "float")
        self.output_vector = tf.zeros([batch_size, class_count], "float")
        self.write_weighting = tf.zeros([batch_size, memory_locations])
        self.read_weightings = tf.zeros([batch_size, read_vector_count, memory_locations])
        self.precedence = tf.zeros([batch_size, memory_locations])
        self.usage = tf.zeros([batch_size, memory_locations])
        self.lstm_states = build_zero_states(node_counts, batch_size)

    def to_tuple(self):
        return (self.read_vectors,
                self.memory,
                self.links,
                self.output_vector,
                self.precedence,
                self.write_weighting,
                self.read_weightings,
                self.usage,
                self.lstm_states)

    def load_tuple(self, tup):
        self.read_vectors = tup[0]
        self.memory = tup[1]
        self.links = tup[2]
        self.output_vector = tup[3]
        self.precedence = tup[4]
        self.write_weighting = tup[5]
        self.read_weightings = tup[6]
        self.usage = tup[7]
        self.lstm_states = tup[8]

def build_zero_states(node_counts, batch_size):
    states = []
    for x in node_counts[:-1]:
        states.append(tf.contrib.rnn.BasicLSTMCell(x, state_is_tuple=True).zero_state(batch_size, tf.float32))
    return states