import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

import seaborn as sns

from datetime import datetime

data_veh = pd.read_csv('D:\craigslistVehicles.csv')


class vehicle(object):
    pass

    def remove_cols(self, data_veh_post):

        start = datetime.now()

        # print(data_veh.columns)

        data_veh_post = data_veh.drop(
            ['url', 'city', 'city_url', 'image_url', 'make', 'VIN', 'lat', 'long', 'title_status', 'drive', 'desc',
             'size'], axis=1)

        # print(data_veh_post.columns)

        self.data_veh_post = data_veh_post

        print('time taken', datetime.now() - start)

    def convert_nominal(self, data_nullify, data):

        start = datetime.now()

        # print(self.data_veh_post)

        self.data = self.data_veh_post

        # data=pd.get_dummies(data,columns=['manufacturer','fuel','transmission','paint_color','city','type'])

        print(self.data.columns)

        print(self.data.shape)

        # print(data.sample)

        # print(data.describe)

        # print(data.isnull().sum())

        self.data_nullify = self.data.isnull().sum() * 100 / self.data.shape[0]

        # print(self.data_drive.shape)

        # print(self.data_nullify)

        print('time taken', datetime.now() - start)

    def nullify(self, data_drive):

        start = datetime.now()

        self.data_drive = self.data

        test_dat = self.data_drive

        # self.data_drive=self.data_drive.dropna()

        # print(self.data_drive.isnull().sum(),self.data_drive.shape)

        self.data_drive['condition'].fillna(self.data_drive['condition'].mode()[0], inplace=True)

        self.data_drive['cylinders'].fillna(self.data_drive['cylinders'].mode()[0], inplace=True)

        self.data_drive['paint_color'].fillna(self.data_drive['paint_color'].mode()[0], inplace=True)

        self.data_drive['type'].fillna(self.data_drive['type'].mode()[0], inplace=True)

        self.data_drive.dropna(inplace=True)

        # print(self.data_drive.isnull().sum(),self.data_drive.shape)

        print('time taken', datetime.now() - start)

    def corr(self):

        start = datetime.now()

        correl = self.data_drive.select_dtypes(include=[np.number])

        corr = correl.corr()

        sns.heatmap(corr)

        plt.show()

        # print(corr['price'].sort_values(ascending=False))

        print('time taken', datetime.now() - start)

    def anova(self):

        start = datetime.now()

        self.data_drive['fuel'] = self.data_drive['fuel'].astype('category').cat.codes

        self.data_drive['condition'] = self.data_drive['condition'].astype('category').cat.codes

        self.data_drive['cylinders'] = self.data_drive['cylinders'].astype('category').cat.codes

        self.data_drive['fuel'] = self.data_drive['fuel'].astype('category').cat.codes

        self.data_drive['transmission'] = self.data_drive['transmission'].astype('category').cat.codes

        self.data_drive['type'] = self.data_drive['type'].astype('category').cat.codes

        self.data_drive['paint_color'] = self.data_drive['paint_color'].astype('category').cat.codes

        # print(self.data_drive[self.data_drive.columns[:]].corr()['price'][:])

        print('time taken', datetime.now() - start)

    def removecol(self):

        start = datetime.now()

        self.data_drive.drop(['transmission', 'type', 'paint_color'], axis=1, inplace=True)

        # print(self.data_drive.columns)

        print('time taken', datetime.now() - start)

    def labelencode(self):

        start = datetime.now()

        test = LabelEncoder()

        self.data_drive['condition'] = test.fit_transform(self.data_drive['condition'])

        self.data_drive['cylinders'] = test.fit_transform(self.data_drive['cylinders'])

        # print(self.data_drive.sample)

        print('time taken', datetime.now() - start)

    def onehotencode(self):

        start = datetime.now()

        # test=OneHotEncoder()

        self.data_drive = pd.get_dummies(self.data_drive, columns=['fuel', 'manufacturer'])

        # self.data_drive['transmission']=pd.get_dummies(self.data_drive['transmission'])

        # self.data_drive['paint_color']=pd.get_dummies(self.data_drive['paint_color'])

        # self.data_drive['type']=pd.get_dummies(self.data_drive['type'])

        print(self.data_drive.head())

        print('time taken', datetime.now() - start)

    def standscal(self, x, y):

        start = datetime.now()

        from sklearn.preprocessing import StandardScaler

        scale = StandardScaler()

        self.x = self.data_drive.drop('price', axis=1)

        self.y = self.data_drive['price']

        self.x = scale.fit_transform(self.x)

        # print(self.x)

        print('time taken', datetime.now() - start)

    def split(self, x_train, x_test, y_train, y_test):

        start = datetime.now()

        from sklearn.model_selection import train_test_split

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.3)

        # print(self.x_train.shape)

        # print(self.x_test.shape)

        print('time taken', datetime.now() - start)

    def metric_method(self, pred, y_test):
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        print('mean absolute error value is', mean_absolute_error(pred, y_test))
        print('mean squared error', mean_squared_error(pred, y_test))
        print('r2 value ', r2_score(pred, y_test))

    def randomforest_minibatch(self, batch_size, x_train_mini, y_train_mini):
        from sklearn.ensemble import RandomForestRegressor
        start = datetime.now()

        for i in range(0, len(self.x_train), batch_size):
            print(len(self.x_train))

            if i > len(self.x_train):
                # print('hello world')

                break

            else:
                # print('its else part')
                # print('shape of x_train {}'.format(self.x_train.shape[0]))
                self.x_train_mini = self.x_train[i:i + batch_size]

                self.y_train_mini = self.y_train[i:i + batch_size]
                test = RandomForestRegressor(max_depth=4, max_features='auto', min_samples_leaf=50, n_estimators=100,
                                             random_state=200)

                test.fit(self.x_train_mini, self.y_train_mini)
        pred = test.predict(self.x_test)

        y_t = self.y_test

        # print(y_t)
        print('length y_test {}'.format(len(y_t)))

        # print((pred - y_t))
        # print('x_train_mini {}'.format(len(self.x_train_mini), len(self.y_train_mini)))
        self.metric_method(pred, y_t)
        print('time taken', datetime.now() - start)

    def Decisiontree_minibatch(self, batch_size, x_train_mini, y_train_mini):
        from sklearn.tree import DecisionTreeRegressor
        start = datetime.now()

        for i in range(0, len(self.x_train), batch_size):
            print(len(self.x_train))

            if i > len(self.x_train):
                # print('hello world')

                break

            else:
                # print('its else part')
                # print('shape of x_train {}'.format(self.x_train.shape[0]))
                self.x_train_mini = self.x_train[i:i + batch_size]

                self.y_train_mini = self.y_train[i:i + batch_size]
                test = DecisionTreeRegressor(max_depth=4, max_features='auto', min_samples_leaf=50,
                                             random_state=200)

                test.fit(self.x_train_mini, self.y_train_mini)
        pred = test.predict(self.x_test)

        y_t = self.y_test

        # print(y_t)
        print('length y_test {}'.format(len(y_t)))

        # print((pred - y_t))
        # print('x_train_mini {}'.format(len(self.x_train_mini), len(self.y_train_mini)))
        self.metric_method(pred, y_t)
        print('time taken', datetime.now() - start)


if __name__ == '__main__':
    my_veh = vehicle()

    my_veh.remove_cols('data_veh_post')

    my_veh.convert_nominal('data_nullify', 'data')

    my_veh.nullify('data_drive')

    my_veh.corr()

    my_veh.anova()

    my_veh.removecol()

    my_veh.labelencode()

    my_veh.onehotencode()

    my_veh.standscal('x', 'y')

    my_veh.split('x_train', 'x_test', 'y_train', 'y_test')

    my_veh.randomforest_minibatch(80000, 'x_train_mini', 'y_train_mini')
    my_veh.Decisiontree_minibatch(80000, 'x_train_mini', 'y_train_mini')
