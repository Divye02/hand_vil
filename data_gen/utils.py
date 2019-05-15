# TODO: Remove commented out stuff once sure we aren't using it and don't plan
# to ever again
import numpy as np

def action_to_bin(action, num_bins=3):
    if action == -1:
        return 0
    true_bins = num_bins - 2
    val = action + 1
    return int((val * true_bins) // 2) + 1

def multilabel_action_to_bin(action, num_bins=3):
    # intervals = np.linspace(-1, 1, true_bins + 1)
    # indices = np.digitize(action, intervals) - 1
    return np.array([action_to_bin(a, num_bins) for a in action])

def bin_to_action(bin_idx, num_bins=3):
    true_bins = num_bins - 2
    interval_length = 2 / true_bins
    interval_offset = interval_length / 2
    
    if bin_idx == 0:
        return -1
    if bin_idx == num_bins - 1:
        return 1
    return (bin_idx - 1) * interval_length + interval_offset - 1
    

def multilabel_bin_to_action(bin_indices, num_bins=3):
    # intervals = np.linspace(-1, 1, true_bins + 1)
    # actions = np.array(get_actions(intervals, bin_indices, num_actions=len(bin_indices)))
    return np.array([bin_to_action(b, num_bins) for b in bin_indices])

# def onehot_action_to_bin(action, num_bins=3):
#     shape = [num_bins for i in range(len(action))]
#     shape = tuple(shape)
#     action_b = np.zeros(shape)
#     intervals = np.linspace(-1, 1, num_bins + 1)
#     indices = np.digitize(action, intervals) - 1
#     index = tuple(indices)
#     action_b[index] = 1
#     pass
#
# def onehot_bin_to_action(bin, num_bins=3):
#     shape = [num_bins for i in range(len(action))]
#     shape = tuple(shape)
#     pass

def get_actions(intervals, indices, num_actions):
    actions = []
    for i in range(num_actions):
        index_int = indices[i]
        actions.append((intervals[index_int] + intervals[index_int + 1])/2)
    return actions
