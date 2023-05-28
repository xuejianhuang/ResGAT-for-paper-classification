from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from imblearn.under_sampling import RandomUnderSampler

def pca_components_analysis(n, x_train, x_test):
    pca = PCA(n_components=n)
    print("Dimensions before dimensionality reduction：{}，{}".format(x_train.shape,x_test.shape))
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)
    print("Dimensions after dimensionality reduction：{}，:{}".format(x_train.shape,x_test.shape))


    return x_train,x_test


def grid_search(estimator, param_grid, cv, x_train, x_test, y_train, y_test):
    gridsearch = GridSearchCV(estimator, param_grid, cv)
    gridsearch.fit(x_train, y_train)
    score = gridsearch.score(x_test, y_test)
    print("Accuracy:\n", score)
    print("best parameters：\n", gridsearch.best_params_)

def under_sampling(x,y):
    print("Dimensions before downsamplingx：{}，y:{}".format(x.shape,y.shape))
    rus = RandomUnderSampler(random_state=0)
    x_resampled, y_resampled = rus.fit_resample(x, y)
    print("Dimensions after downsamplingx：{}，y:{}".format(x_resampled.shape, y_resampled.shape))
    return x_resampled,y_resampled
