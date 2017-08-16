import tensorflow as tf

# A class that wraps a batch of interface vectors in order to provide the specific factors from them
class InterfaceWrapper(object):
    def __init__(self, vector, read_vector_count = 1, memory_vector_size = 8):
        self.vector = vector
        self.read_vector_count = read_vector_count
        self.memory_vector_size = memory_vector_size

    # The key that the controller network will use as a lookup for read addressing
    def read_key(self, index):
        return self.vector[:, index * self.memory_vector_size : (index + 1) * self.memory_vector_size]

    # The strength of the read, essentially just weights the read vector
    def read_strength(self, index):
        start = self.read_vector_count * self.memory_vector_size
        return oneplus(self.vector[:, start + index])

    # The key that the controller network will use as a lookup for write addressing
    def write_key(self):
        start = self.read_vector_count * (self.memory_vector_size + 1)
        return self.vector[:, start : start + self.memory_vector_size]

    # The vector that well be "Subtracted" from the written memory locations
    def erase_vector(self):
        start = self.read_vector_count * self.memory_vector_size + self.memory_vector_size + self.read_vector_count
        return tf.sigmoid(self.vector[:, start : start + self.memory_vector_size])

    # The vector that will be "Added" to the written memory locations
    def write_vector(self):
        start = self.read_vector_count * self.memory_vector_size + self.read_vector_count + (self.memory_vector_size * 2)
        return self.vector[:, start : start + self.memory_vector_size]

    # The amount that memory locations should be "Freed" after reading from them
    # If a memory location is freed, it is more likely to be overwritten in the usage weighting
    def free_gate(self, index):
        start = self.read_vector_count * self.memory_vector_size + self.read_vector_count + (self.memory_vector_size * 3)
        return tf.sigmoid(self.vector[:, start + index])

    # The different modes that should be used when calculating the read weighting
    # 1) Forward linking
    # 2) Content lookup
    # 3) Backward linking
    def read_modes(self, index):
        start = self.read_vector_count * self.memory_vector_size + (self.read_vector_count * 2) + (self.memory_vector_size * 3)
        return tf.nn.softmax(self.vector[:, start + ((index - 1) * 3) : start + (index * 3)])

    # Controls the degree to which memory writing is controlled by usage vs content lookup
    def allocation_gate(self):
        start = self.read_vector_count * self.memory_vector_size + (self.read_vector_count * 5) + (self.memory_vector_size * 3)
        return tf.sigmoid(self.vector[:, start])

    # Limits the weighting of the write
    def write_gate(self):
        start = self.read_vector_count * self.memory_vector_size + (self.read_vector_count * 5) + (self.memory_vector_size * 3)
        return tf.sigmoid(self.vector[:, start + 1])

    # Limits the strength of the write
    def write_strength(self):
        start = self.read_vector_count * self.memory_vector_size + (self.read_vector_count * 5) + (self.memory_vector_size * 3)
        return oneplus(self.vector[:, start + 2])

# Restricts a vector to the [1, +inf) range
def oneplus(x):
    return 1 + tf.log(1 + tf.exp(x))
