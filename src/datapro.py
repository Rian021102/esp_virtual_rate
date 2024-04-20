from sklearn.impute import SimpleImputer

class dataprocess():
    def __init__(self,X_train,X_test):
        self.X_train = X_train
        self.X_test = X_test
    def process_train(self):
        #fill missing values with imputing
        imputer = SimpleImputer(strategy='median')
        imputer.fit(self.X_train)
        self.X_train = imputer.transform(self.X_train)
        return self.X_train
    def process_test(self):
        #fill missing values with imputing
        imputer = SimpleImputer(strategy='mean')
        imputer.fit(self.X_test)
        self.X_test = imputer.transform(self.X_test)
        return self.X_test
    