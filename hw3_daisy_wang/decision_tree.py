import random
import numpy as np
import pandas as pd

class DecisionNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature          # Index of the feature used for splitting
        self.threshold = threshold      # Threshold value for the split
        self.left = left                # Left subtree (for values <= threshold)
        self.right = right              # Right subtree (for values > threshold)
        self.value = value              # Value if it's a leaf node (e.g., class label)

def gini_index(groups, classes):
    # Calculate Gini index for a split
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        gini += (1.0 - score) * (size / n_instances)
    return gini

def test_split(feature_index, threshold, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[feature_index] <= threshold:
            left.append(row)
        else:
            right.append(row)
    return left, right

def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}

def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)

def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root

def predict(node, row):
    if row[node['index']] <= node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

# Load the data and preprocess
data = pd.read_csv('data.csv')

# Encode categorical variables using label encoding for simplicity
for col in data.select_dtypes(include=['object']).columns:
    data[col] = data[col].astype('category').cat.codes

# Drop any non-feature columns like 'person ID'
data = data.drop(columns=['person ID'])
dataset = data.values.tolist()

# Train-test split
random.shuffle(dataset)
train_size = int(0.7 * len(dataset))
train_data = dataset[:train_size]
test_data = dataset[train_size:]

# Build and evaluate the tree
tree = build_tree(train_data, max_depth=5, min_size=10)
predictions = [predict(tree, row) for row in test_data]
actual = [row[-1] for row in test_data]
print(predictions)
print(actual)
# Calculate accuracy
accuracy = sum(1 for i in range(len(actual)) if actual[i] == predictions[i]) / len(actual)
print(f"Accuracy: {accuracy:.2f}")
