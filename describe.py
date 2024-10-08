'''
usage: python3 describe.py <dataset>
example: python3 describe.py datasets/dataset_train.csv


describe.py is a program that computes the following statistics for each feature in the dataset:
- count - number of non-NA/null observations
- mean - average value of the dataset
- standard deviation - shows how much variation from the mean exists
- minimum value 
- 25th percentile - value below which 25% of the observations fall
- 50th percentile - median value of the dataset - value below which 50% of the observations fall
- 75th percentile - value below which 75% of the observations fall
- maximum value 
Bonus features:
- Unique Count - number of unique values in a dataset
- Mode - most frequent value in a dataset
- Range - difference between the maximum and minimum values
'''

import pandas as pd
import sys


# Remove NaN and not numeric values
def clean_nan(lst):
    return [x for x in lst if isinstance(x, (float, int)) and x == x]

# Count number of non-NA/null observations
def ft_count(lst):
    return len(clean_nan(lst))

# Maximum value in the list
def ft_max(lst):
    max = lst[0]
    for i in range(1, len(lst)):
        if lst[i] > max:
            max = lst[i]
    return max

# Minimum value in the list
def ft_min(lst):
    min = lst[0]
    for i in range(1, len(lst)):
        if lst[i] < min:
            min = lst[i]
    return min

# Mean value of the list
def ft_mean(lst):
    if len(lst) == 0:
        return 0
    lst = clean_nan(lst)
    sum_lst = 0
    for i in lst:
        sum_lst += i
    return sum_lst / len(lst)

# Standard deviation of the list
# measure of how much variation from the mean exists
def ft_std(lst):
    sum_lst = 0
    for i in lst:
        sum_lst += i
    return (sum([(x - ft_mean(lst)) ** 2 for x in lst]) / len(lst)) ** 0.5

# Percentile score
# Meaning p% of the observations are below this value
def ft_percentile(lst, p):
    lst.sort()
    index = (len(lst) - 1) * p
    lower = int(index)
    upper = lower + 1 if lower + 1 < len(lst) else lower
    
    if upper == lower:
        return lst[lower]
    else:
        return lst[lower] + (index - lower) * (lst[upper] - lst[lower])

def ft_25(lst):
    return ft_percentile(lst, 0.25)

def ft_50(lst):
    return ft_percentile(lst, 0.50)

def ft_75(lst):
    return ft_percentile(lst, 0.75)


# Unique values in the list
def ft_unique(lst):
    return len(set(lst))

# mode - most frequent value in the list
def ft_mode(lst):
    return max(set(lst), key = lst.count)

# Range - difference between the maximum and minimum values
def ft_range(lst):
    return ft_max(lst) - ft_min(lst)


# Create dataframe with describe values
def describe(features):
    funcs = [{"count": ft_count}, {"mean": ft_mean}, {"std": ft_std}, {"min": ft_min}, 
             {"25%": ft_25}, {"50%": ft_50}, {"75%": ft_75}, {"max": ft_max}]
    describe = pd.DataFrame()
    for i in range(len(features.columns)):
        col = clean_nan(features.iloc[:,i].tolist())
        for func in funcs:
            for key, value in func.items():
                describe.loc[key, features.columns[i]] = value(col)
    return describe

def describe_bonus(features):
    funcs = [{"count": ft_count}, {"mean": ft_mean}, {"std": ft_std}, {"min": ft_min}, 
             {"25%": ft_25}, {"50%": ft_50}, {"75%": ft_75}, {"max": ft_max},
             {"unique": ft_unique}, {"mode": ft_mode}, {"range": ft_range}]
    describe = pd.DataFrame()
    for i in range(len(features.columns)):
        col = clean_nan(features.iloc[:,i].tolist())
        for func in funcs:
            for key, value in func.items():
                describe.loc[key, features.columns[i]] = value(col)
    return describe


if __name__ == '__main__':

    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("Usage: python3 describe.py <dataset> <bonus>")
        sys.exit(1)

    dataset = sys.argv[1]
    is_bonus = sys.argv[2] if len(sys.argv) > 2 else False

    df = pd.read_csv(dataset)
    features = df[["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", 
                   "Divination", "Muggle Studies", "Ancient Runes", "History of Magic", 
                   "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"]]
    labels = ["Feature " + str(i+1) for i in range(len(features.columns))] 
    features.columns = labels
    
    pd.set_option('display.max_rows', None)

    if is_bonus:
        print(describe_bonus(features))
    else:
        print(describe(features))
    
