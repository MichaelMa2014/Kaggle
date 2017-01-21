import pandas


def load_train_user_data():
    """
    读入训练集数据
    :rtype: pandas.DataFrame
    """
    return pandas.read_csv('data/train.csv', names=['CONS_NO', 'LABEL'])


def load_test_user_data():
    """
    读入测试集数据
    :rtype: pandas.DataFrame
    """
    return pandas.read_csv('data/test_csv')