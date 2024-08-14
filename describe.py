import pandas as pd

# Remove NaN and not numeric values
def clean_nan(lst):
    for i in lst:
        if i != i or not isinstance(i, (float, int)): 
            lst.remove(i)
    return lst

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
    # print(sum_lst)
    # print(len(lst))
    return sum_lst / len(lst)

# Standard deviation of the observations
def ft_std(lst):
    return (sum([(x - ft_mean(lst)) ** 2 for x in lst]) / len(lst)) ** 0.5

# 25-percentile score
# Meaning 25% of the observations are below this value
def ft_25(lst):
    lst.sort()
    return lst[int(len(lst) * 0.25)]

# 50-percentile score
# Meaning 50% of the observations are below this value
def ft_50(lst):
    #clean_nan(lst)
    lst.sort()
    return lst[int(len(lst) * 0.5)]

# 75-percentile score
# Meaning 75% of the observations are below this value
def ft_75(lst):
    lst.sort()
    return lst[int(len(lst) * 0.75)]

# Create dataframe with describe values
def describe(features):
    funcs = [{"count": ft_count}, {"mean": ft_mean}, {"std": ft_std}, {"min": ft_min}, 
             {"25%": ft_25}, {"50%": ft_50}, {"75%": ft_75}, {"max": ft_max}]
    describe = pd.DataFrame()
    for i in range(len(features.columns)):
        col = features.iloc[:,i].tolist()
        for func in funcs:
            for key, value in func.items():
                describe.loc[key, features.columns[i]] = value(col)
    return describe

if __name__ == '__main__':
    df = pd.read_csv('datasets/dataset_train.csv')
    features = df[["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", 
                   "Divination", "Muggle Studies", "Ancient Runes", "History of Magic", 
                   "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"]]
    labels = ["Feature " + str(i+1) for i in range(len(features.columns))] 
    features.columns = labels
    print(describe(features))
    # print(features.describe()) to check with pandas describe function
