import tensorflow as tf

# A class that wraps a batch of interface vectors in order to provide the specific factors from them
class InterfaceWrapper(object):
    def __init__(self, vector, read_vector_count = 1, memory_vector_size = 8):
        self.vector = vector
        self.read_vector_count = read_vector_count
        self.memory_vector_size = memory_vector_size

    def read_key(self, index):
        return self.vector[:, index * self.memory_vector_size : (index + 1) * self.memory_vector_size]

    def read_strength(self, index):
        start = self.read_vector_count * self.memory_vector_size
        return oneplus(self.vector[:, start + index])

    def write_key(self):
        start = self.read_vector_count * (self.memory_vector_size + 1)
        return self.vector[:, start : start + self.memory_vector_size]

    def erase_vector(self):
        start = self.read_vector_count * self.memory_vector_size + self.memory_vector_size + self.read_vector_count
        return tf.sigmoid(self.vector[:, start : start + self.memory_vector_size])

    def write_vector(self):
        start = self.read_vector_count * self.memory_vector_size + self.read_vector_count + (self.memory_vector_size * 2)
        return self.vector[:, start : start + self.memory_vector_size]

    def free_gate(self, index):
        start = self.read_vector_count * self.memory_vector_size + self.read_vector_count + (self.memory_vector_size * 3)
        return tf.sigmoid(self.vector[:, start + index])

    def read_modes(self, index):
        start = self.read_vector_count * self.memory_vector_size + (self.read_vector_count * 2) + (self.memory_vector_size * 3)
        return tf.nn.softmax(self.vector[:, start + ((index - 1) * 3) : start + (index * 3)])

    def allocation_gate(self):
        start = self.read_vector_count * self.memory_vector_size + (self.read_vector_count * 5) + (self.memory_vector_size * 3)
        return tf.sigmoid(self.vector[:, start])

    def write_gate(self):
        start = self.read_vector_count * self.memory_vector_size + (self.read_vector_count * 5) + (self.memory_vector_size * 3)
        return tf.sigmoid(self.vector[:, start + 1])

    def write_strength(self):
        start = self.read_vector_count * self.memory_vector_size + (self.read_vector_count * 5) + (self.memory_vector_size * 3)
        return oneplus(self.vector[:, start + 2])

def oneplus(x):
    return 1 + tf.log(1 + tf.exp(x))
