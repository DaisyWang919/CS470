import pandas as pd
import random

def generate_train_test_ids(data_path, train_ids_path, test_ids_path, train_ratio=0.7):
    # Load the dataset
    data = pd.read_csv(data_path)
    
    # Extract the person IDs
    person_ids = data['person ID'].astype(str).tolist()
    
    # Shuffle the IDs randomly
    random.shuffle(person_ids)
    
    # Split IDs into training and testing sets
    split_index = int(train_ratio * len(person_ids))
    train_ids = person_ids[:split_index]
    test_ids = person_ids[split_index:]
    
    # Write training IDs to the file
    with open(train_ids_path, 'w') as train_file:
        for person_id in train_ids:
            train_file.write(person_id + '\n')
    
    # Write testing IDs to the file
    with open(test_ids_path, 'w') as test_file:
        for person_id in test_ids:
            test_file.write(person_id + '\n')
    
    print(f"Training IDs saved to {train_ids_path}")
    print(f"Testing IDs saved to {test_ids_path}")

# Example usage
generate_train_test_ids('data.csv', 'para2_file.txt', 'para3_file.txt')
