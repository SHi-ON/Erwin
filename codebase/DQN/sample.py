class Sample(object):

    def __init__(self, df, yrs, r_col):
        self.df = df
        self.yrs = yrs
        self.r_col = r_col

    # return ndarray version of the data frame
    def _nd_array(self):
        return self.df.values

    # return number of epochs exist in dataset
    def get_epochs(self):
        return self.df.loc[:, 'V1'].value_counts().to_dict()[self.yrs]

    def is_over(self, ep):
        return ep == self.yrs - 1

    def get_reward(self, epoch, index):
        values = self._nd_array()
        row = epoch * self.yrs
        return values[row + index, self.r_col]

    def get_init_sample(self, epoch, start=1, end=19):
        values = self._nd_array()
        row = epoch * self.yrs
        return values[row, start:end].reshape((1, -1))

    def get_sample(self, epoch, index, start=1, end=19):
        values = self._nd_array()
        row = epoch * self.yrs
        return values[row + index, start:end].reshape((1, -1))
