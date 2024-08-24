import pandas as pd
import sys
# descruibe.py is a program that computes the following statistics for each feature in the dataset:
# - count - number of non-NA/null observations
# - mean - average value of the dataset
# - standard deviation - shows how much variation from the mean exists
# - minimum value 
# - 25th percentile - value below which 25% of the observations fall
# - 50th percentile - median value of the dataset - value below which 50% of the observations fall
# - 75th percentile - value below which 75% of the observations fall
# - maximum value 
# Bonus features:
# - Duplicate Count - number of duplicate values in a dataset
# - Z-score - measure of how many standard deviations an element is from the mean - to check for outliers
# - Unique Count - number of unique values in a dataset
# - Mode - most frequent value in a dataset
# - Skewness - measure of the asymmetry of the probability distribution of a real-valued random variable about its mean
# - Range - difference between the maximum and minimum values

# The program should be run as follows:
# python describe.py datasets/dataset_train.csv


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

if __name__ == '__main__':
    dataset = sys.argv[1]
    df = pd.read_csv(dataset)
    features = df[["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", 
                   "Divination", "Muggle Studies", "Ancient Runes", "History of Magic", 
                   "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"]]
    labels = ["Feature " + str(i+1) for i in range(len(features.columns))] 
    features.columns = labels
    
    pd.set_option('display.max_rows', None)
    print(describe(features))
    
    # to check with pandas describe function
    # print(features.describe()) 

