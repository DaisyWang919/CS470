import random
import numpy as np
import pandas as pd
import argparse

class DecisionNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def gini_index(groups, classes):
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
    del node['groups']
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
        split(node['left'], max_depth, min_size, depth + 1)
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth + 1)

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

def main(data_path, train_ids_path, test_ids_path, output_path):
    # Load data
    data = pd.read_csv(data_path)

    # Encode categorical variables using label encoding
    for col in data.select_dtypes(include=['object']).columns:
        if col != 'Has heart disease? (Prediction Target)':
            data[col] = data[col].astype('category').cat.codes

    # Map the target variable
    data['Has heart disease? (Prediction Target)'] = data['Has heart disease? (Prediction Target)'].map({'Yes': 1, 'No': 0})

    # Load training and testing IDs
    with open(train_ids_path, 'r') as file:
        train_ids = set(file.read().splitlines())
    with open(test_ids_path, 'r') as file:
        test_ids = set(file.read().splitlines())

    # Split data based on IDs
    train_data = data[data['person ID'].astype(str).isin(train_ids)].drop(columns=['person ID']).values.tolist()
    test_data = data[data['person ID'].astype(str).isin(test_ids)].drop(columns=['person ID']).values.tolist()

    # Build and evaluate the tree
    tree = build_tree(train_data, max_depth=20, min_size=1)
    predictions = [predict(tree, row) for row in test_data]
    actual = [row[-1] for row in test_data]

    # Save predictions to output file
    with open(output_path, 'w') as file:
        for i, row in enumerate(test_data):
            person_id = str(row[0])
            prediction = 'yes' if predictions[i] == 1 else 'no'
            file.write(f"{person_id} {prediction}\n")

    # Print evaluation metrics
    accuracy = sum(1 for i in range(len(actual)) if actual[i] == predictions[i]) / len(actual)
    print(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decision Tree Implementation")
    parser.add_argument("data_path", help="Path to the input dataset file")
    parser.add_argument("train_ids_path", help="Path to the training IDs file")
    parser.add_argument("test_ids_path", help="Path to the testing IDs file")
    parser.add_argument("output_path", help="Path to save the prediction output file")

    args = parser.parse_args()
    main(args.data_path, args.train_ids_path, args.test_ids_path, args.output_path)
