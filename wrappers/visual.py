
def draw_learning_curve(estimator, X, y, scoring="accuracy", cv=5, lower=0.75, higher=0.95):

    from sklearn.model_selection import learning_curve
    import matplotlib.pyplot as plt
    import numpy as np

    train_sizes, train_scores, test_scores = \
        learning_curve(estimator=estimator, X=X, y=y,
                       train_sizes=np.linspace(0.1, 1.0, 10),
                       scoring=scoring, cv=cv, n_jobs=-1)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean,
             color='blue', marker='o',
             markersize=5, label='training accuracy')

    plt.fill_between(train_sizes,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, color='blue')

    plt.plot(train_sizes, test_mean,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label='validation accuracy')

    plt.fill_between(train_sizes,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')

    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([lower, higher])
    plt.tight_layout()
    plt.show()


def corrplot(df):

    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt

    sns.set(style="white")
    corr = df.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, square=True, ax=ax)
    sns.plt.show()


def hist_compare(train, test, title, xlabel, bins):
    """
    show normalized histograms for train and test parameter
    :return: show plot, mean/std/range statistics
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    pal = sns.color_palette()
    plt.figure(figsize=(15, 10))
    plt.hist(train, bins=bins, range=[0, bins], color=pal[2], normed=True, label='train')
    plt.hist(test, bins=bins, range=[0, bins], color=pal[1], normed=True, alpha=0.5, label='test')
    plt.title(title, fontsize=15)
    plt.legend()
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('Probability', fontsize=15)
    plt.show()
    print('mean-train {:.2f} std-train {:.2f} mean-test {:.2f} std-test {:.2f} max-train {:.2f} max-test {:.2f}'.format(
        train.mean(),train.std(), test.mean(), test.std(), train.max(), test.max()))


def word_cloud(df):
    """
    simple word cloud
    :param df: pandas Series
    :return: show plot
    """
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    cloud = WordCloud(width=1440, height=1080).generate(" ".join(df.astype(str)))
    plt.figure(figsize=(20, 15))
    plt.imshow(cloud)
    plt.axis('off')
    plt.show()
