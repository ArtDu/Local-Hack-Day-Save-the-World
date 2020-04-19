import joblib


def dummy_fun(doc):
    return doc


if __name__ == '__main__':

    joblib.dump(dummy_fun, './models/dummy_fun.sav')
