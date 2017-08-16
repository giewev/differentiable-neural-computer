import numpy as np

# Loads and vectorizes a babi dataset file
# Also creates a corresponding target sequence
def load_babi_file(path):
    data = read_babi_file(path)
    ids, data = vectorize_babi_file(data)
    data, targets = build_babi_targets(data, ids)
    sequence_lengths = [len(x) for x in data]
    maximum_sequence_length = max(sequence_lengths)
    data = [zero_pad_sequence(x, maximum_sequence_length) for x in data]
    targets = [zero_pad_sequence(x, maximum_sequence_length) for x in targets]
    return list(zip(data, targets))

# Reads a babi file and splits it into seperate stories and words
def read_babi_file(path):
    with open(path) as f:
        lines = f.readlines()
        lines = [x.replace('\t', ' ').replace('\n', '').split(' ') for x in lines]
        lines = [[x for x in y if x not in ["", " ", "\n"]] for y in lines]
        stories = []
        for x in lines:
            if int(x[0]) == 1:
                stories.append([])
            stories[-1].append(x)
        return stories

# Checks if a string is numeric
def is_numeric(string):
    try:
        int(string)
        return True
    except:
        return False

# builds a lookup for the one hot indices in the vectorization of a story
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

# Takes a target vectorization and simplifies it by removing any values that are not possible answers
def simplify_vectorization(stories):
    targeted_stories = []
    target_ids = dict()

    # Find all of the indices that the story is actually using
    for story in stories:
        for ind, word in enumerate(story):
            if not np.all(word == 0):
                word_id = np.argmax(word)
                if word_id not in target_ids:
                    target_ids[word_id] = len(target_ids)

    # Generate new vectors for the targets based on the new IDs
    for story in stories:
        targets = []
        for ind, word in enumerate(story):
            target = np.zeros(shape = (len(target_ids),), dtype = float)
            if not np.all(word == 0):
                target = one_hot(len(target_ids), target_ids[np.argmax(word)])
            targets.append(target)
        targeted_stories.append(targets)
    return targeted_stories

# Makes a one hot float vector 
def one_hot(size, index):
    a = np.zeros(shape = (size,), dtype = float)
    a[index] = 1
    return a

# Converts the words in the story to one hot vectors, and creates a corresponding target story
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

# Removes the answers from the input stories, and creates a corresponding target vector story
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
    targeted_stories = simplify_vectorization(targeted_stories)
    return (stories, targeted_stories)

# Pads a sequence out with 0 vectors until it is the given length
def zero_pad_sequence(sequence, length):
    while len(sequence) < length:
        sequence.append(np.zeros(shape = np.shape(sequence[0]), dtype = float))
    return sequence