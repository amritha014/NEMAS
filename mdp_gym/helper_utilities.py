import numpy as np

def flatten_state(state, size, state_space):
    """Flatten state (x,y) into a one hot vector"""
    idx = size * state[0] + state[1]
    one_hot = np.zeros(state_space)
    one_hot[idx] = 1
    return one_hot


def unflatten_state(onehot, size, has_absorbing_state):
    """Unflatten a one hot vector into a (x,y) pair"""
    # if has_absorbing_state:
    #     onehot = onehot[:-1]
    onehot = onehot.reshape(size, size)
    x = onehot.argmax(0).max()
    y = onehot.argmax(1).max()
    return (x, y)


