# %%
from main import (
    proceed_problem1,
    proceed_problem2,
    proceed_problem3,
    proceed_problem4
    )
from main import NewsCategory, NEWS_CATEGORY_DATASET, BATCH_COUNT, svm
import numpy as np

# %%
if __name__ == "__main__":
    for i in range(0, len(NEWS_CATEGORY_DATASET), BATCH_COUNT):
        X = []
        y = []

        t = 0
        for c in NEWS_CATEGORY_DATASET[i:i + BATCH_COUNT]:
            X.append(proceed_problem2(c['headline']))
            y.append(NewsCategory[c['category']].value)

        X = np.asarray(X)
        y = np.asarray(y)

        for x, y_ in zip(X, y):
            x = np.asarray([x])
            res = svm.predict(x)
            print(NewsCategory(res).name, " | ", NewsCategory(y_).name)
            if(res == y_):
                t = t + 1

        print(t / BATCH_COUNT)


# %%
