import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __name__ == '__main__':
    # Which Hogwarts course has a homogeneous score distribution between all four houses?
    print("hello")
    try:
        df = pd.read_csv("dataset_train.csv")
        print("Loading datasets")
    except OSError as err:
        print("OS error: {0}".format(err))
    else:
        features1 = ['Index', 'Hogwarts House', 'First Name', 'Last Name',
                        'Birthday', 'Best Hand', 'Arithmancy', 'Astronomy', 'Herbology',
                        'Defense Against the Dark Arts', 'Divination', 'Muggle Studies',
                        'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions',
                        'Care of Magical Creatures', 'Charms', 'Flying']
        categorical = ['Hogwarts House']
        features = ['Transfiguration']

        print(df)
        sns.histplot(data=df, x='History of Magic', hue='Hogwarts House', element = 'step')
        plt.show()
