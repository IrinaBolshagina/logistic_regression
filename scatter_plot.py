import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __name__ == '__main__':
    # What are the two features that are similar ?
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

        sns.scatterplot(data=df, x='Charms', y='Care of Magical Creatures', hue = 'Hogwarts House')
        plt.show()
