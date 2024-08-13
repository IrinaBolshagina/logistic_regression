import pandas as pd

# Count number of non-NA/null observations
def ft_count(lst):
    for i in lst:
        if i == 'NaN' or not i:
            lst.remove(i)
    return len(lst)

# Maximum value in the list
def ft_max(lst):
    max = lst[0]
    for i in range(1, len(lst)):
        if i > max:
            max = i
    return max

# Minimum value in the list
def ft_min(lst):
    min = lst[0]
    for i in range(1, len(lst)):
        if i < min:
            min = i
    return min

# Mean value of the list
def ft_mean(lst):
    return sum(lst) / len(lst)

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
    lst.sort()
    return lst[int(len(lst) * 0.5)]

# 75-percentile score
# Meaning 75% of the observations are below this value
def ft_75(lst):
    lst.sort()
    return lst[int(len(lst) * 0.75)]

def describe(df):
    for i in range(df.shape[1]):
        col = features.loc[i].tolist()
        print("Feature " + str(i+1))
        print("Count: " + str(ft_count(col)))
        print("Mean: " + str(ft_mean(col)))

if __name__ == '__main__':
    df = pd.read_csv('datasets/dataset_train.csv')
    features = df[["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", "Divination", "Muggle Studies", "Ancient Runes", "History of Magic", "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"]]
    labels = ["Feature " + str(i+1) for i in range(len(features.columns))] 
    features.columns = labels
    print(features)
    describe(df)
    describe = pd.DataFrame()
    describe = pd.DataFrame()
    for i in range(len(labels)):
        col = features.iloc[:,i].tolist()
        describe[labels[i]] = [ft_count(col), ft_mean(col), ft_std(col), ft_min(col), ft_25(col), ft_50(col), ft_75(col), ft_max(col)]
    print(describe)
