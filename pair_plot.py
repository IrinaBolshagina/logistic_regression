import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __name__ == '__main__':
    # What are the two features that are similar ?
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

        sns.set_theme(style="ticks")
        sns.pairplot(df[['Hogwarts House', 'History of Magic','Charms', 'Flying']], hue='Hogwarts House')
        plt.show()
