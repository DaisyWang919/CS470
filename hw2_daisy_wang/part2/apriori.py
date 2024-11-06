# /* THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE WRITTEN BY OTHER STUDENTS. Daisy Wang */
# Collaboration: Only refer to class materials and Python syntax libraries

import pandas as pd
from itertools import combinations

def call_apriori(input, output, min_support):
    df = pd.read_csv(input)
    transactions = []
    for keywords in df['text_keywords']:
        transaction = set(keywords.split(';'))
        transactions.append(transaction)
    itemsets = apriori(transactions, min_support)
    result = []
    for itemsets_line in itemsets:
        for itemset, support in itemsets_line.items():
            itemset = ' '.join(itemset)
            result.append(f"{itemset} ({support})")
    result.sort(key=lambda x: int(x.split('(')[-1].strip(')')), reverse=True)

    with open(output, 'w') as file:
        for each in result:
            file.write(f"{each}\n")
    
def apriori(transactions, min_support):
    itemsets = {}
    for transaction in transactions:
        for item in transaction:
            key = frozenset([item])
            if key not in itemsets:
                itemsets[key] = 0
            itemsets[key] += 1
    
    for key, value in itemsets.copy().items():
        if value < min_support:
            del itemsets[key]

    k = 1

    while itemsets:
        yield itemsets
        previous_itemsets = set(itemsets.keys())
        new_itemsets = generate_itemsets(previous_itemsets, k)
        new_itemsets = prune_itemsets(new_itemsets, previous_itemsets)
        itemsets = {}
        for itemset in new_itemsets:
            support = count_support(itemset, transactions)
            if support >= min_support:
                itemsets[itemset] = support
        k += 1


def generate_itemsets(k_itemsets, k):
    k1_itemsets = set()
    k_itemsets = sorted(list(k_itemsets))
    for i in range(len(k_itemsets)):
        for j in range(i+1, len(k_itemsets)):
            set1 = list(k_itemsets[i])
            set1.sort()
            set2 = list(k_itemsets[j])
            set2.sort()
            if set1[:-1] == set2[:-1]:
                itemset = frozenset(set1) | frozenset(set2)
                if len(itemset) == k + 1:
                    k1_itemsets.add(itemset)
    return k1_itemsets

def prune_itemsets(itemsets, previous_itemsets):
    prune_itemsets = set()
    for itemset in itemsets:
        frequent = True
        subsets = combinations(itemset, len(itemset) - 1)
        for subset in subsets:
            if frozenset(subset) not in previous_itemsets:
                frequent = False
                break
        if frequent:
            prune_itemsets.add(itemset)
    return prune_itemsets

def count_support(itemset, transactions):
    support = 0
    for transaction in transactions:
        if itemset.issubset(transaction):
            support += 1
    return support

if __name__ == '__main__':
    call_apriori('data.csv', 'output.txt', 262)