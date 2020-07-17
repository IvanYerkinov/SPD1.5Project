import sys
from cmd import Cmd
import pandas as pd


class Data():
    def __init__(self, filename):
        self.dataName = filename
        self.df = pd.read_csv(filename)
        self.df_name = "df"
        self.transformed_data = ""
        self.transformed = 0
        self.kmeans = 0
        self.target = 0
        self.returnList = []
        pass

    def _add(self, str, indent=0):
        for i in range(0, indent):
            str = "    " + str
        self.returnList.append(str)
        return self._add

    def _endblock(self):
        self.returnList.append("")

    def _dataProcesses(self):
        self._add("import pandas as pd")("import numpy as np")("import matplotlib.pyplot as plt")("import seaborn as sns")
        self._add("{} = pd.read_csv({})".format(self.df_name, self.dataName))
        self._endblock()

    def _drop(self, col):  # Col = list of columns
        self._add("{} = {}.drop(columns={})".format(self.df_name, self.df_name, list(col)))
        self._endblock()

    def _target(self, tar):  # Tar = string name of column
        self.target = 1
        self._add("en_target = {}[{}]".format(self.df_name, tar))
        self._endblock()

    def _newdf(self, newname):
        self._add("{} = {}".format(newname, self.df_name))
        self.df_name = newname
        self._endblock()
        self.target = 1

    def _scaleData(self):
        if self.transformed == 0:
            self._add("from sklearn.preprocessing import StandardScaler")
            self.transformed_data = "data_transformed"
            self.transformed = 1

        self._add("scalar = StandardScaler()")("data_transformed = scalar.fit_transform({})".format(self.df_name))
        self._endblock()

    def _kMeans(self, range=10):
        if self.kmeans == 0:
            self._add("import numpy as np")("from scipy.spatial import distance")
            self.kmeans = 1

        self._add("dis = []")("K = range(1, {})".format(range))("for k in K:")
        self._add("km = KMeans(n_clusters=k)", 1)("km.fit({})".format(self.transformed_data), 1)("dis.append(sum(np.min(distance.cdist({}, km.cluster_centers_, 'euclidean'), axis=1)) / {}.shape[0])".format(self.transformed_data, self.transformed_data), 1)
        self._add("plt.plot(K, dis, 'bx-')")("plt.xlabel('k')")("plt.ylabel('Distortion')")("plt.title('The Elbow Method showing the optimal k')")("plt.show()")
        self._endblock()

    def _corre(self):
        self._add("data_corelation = {}.corr()".format(self.df_name))("sns.heatmap(data_corelation)")
        self._endblock()

    def _linearReg(self, col1, col2):
        self._add("from sklearn.model_selection import train_test_split")("X = np.array({}['{}'])".format(self.df_name, col1))("y = np.array({}['{}'])".format(self.df_name, col2))
        self._add("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)")
        self._endblock()
        self._add("from sklearn.linear_model import LinearRegression")("from sklearn.metrics import r2_score")
        self._endblock()
        self._add("reg = LinearRegression()")("X_train = np.array(X_train).reshape(-1, 1)")("y_train = np.array(y_train).reshape(-1, 1)")
        self._endblock()
        self._add("X_test = np.array(X_test).reshape(-1, 1)")("testvals = reg.predict(X_test)")
        self._add("")("print(reg.coef_)")("print(Tax_reg.intercept_)")("print(Tax_reg.score(X_test, y_test))")
        self._endblock()

    def _prepca(self, tar):
        if self.target == 0:
            self._target(tar)
            self._drop(tar)
        self._scaleData()
        self._endblock()

    def _pca(self):
        self._add("from sklearn.decomposition import PCA")("pca = PCA(n_components=2)")("principalComponents = pca.fit_transform({})".format(self.transformed_data))
        self._add("principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])")
        self._add("finalDf = pd.concat([principalDf, en_target], axis=1)")
        self._endblock()

    def _logreg(self, col):
        self._add("feature_cols = {}".format(col))("X = pima[feature_cols]")("y = pima[{}]".format("en_target"))
        self._add("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)")
        self._endblock()
        self._add("from sklearn.linear_model import LogisticRegression")("logreg = LogisticRegression()")("logreg.fit(X_train, y_train)")
        self._endblock()
        self._add("y_pred = logreg.predict(X_test)")


class Shell(Cmd):
    prompt = 'Data Shell: '

    def _(self, c):
        return c.split(" ")

    def __init__(self, filename):
        super().__init__()

        self.data = Data(filename)
        self.data._dataProcesses()

    def do_exit(self, np):
        return True

    def emptyline(self):
        pass

    def do_show(self, np):
        print(self.data.df)

    def do_drop(self, np):
        np = self._(np)
        self.data._drop(np)
        self.data.df.drop(columns=np)

    def do_linearreg(self, np):
        np = self._(np)

        if len(np) < 2:
            print("Please enter two valid column names.")
            return 0

        self.data._linearReg(np[0], np[1])

    def do_kmeans(self, np):
        if self.data.transformed == 0:
            self.data._scaleData()
        self.data._kMeans()

    def do_return(self, np):
        for i in self.data.returnList:
            print(i)
        return True

    def do_new(self, np):
        np = self._(np)

        if len(np) < 1:
            print('Please enter a valid dataframe name')
            return 0
        self.data._newdf(np[0])
        self.data.transformed = 0

    def do_pca(self, np):
        np = self._(np)
        if self.data.target == 0:
            print("Please define a target column")
            return

        self.data._prepca(np[0])
        self.data._pca()

    def do_logreg(self, np):
        np = self._(np)

        if self.data.target == 0:
            print("Please define a target column")
            return
        if len(np) < 1:
            print("Please enter the columns you want to log reg")
            return

        self.data._logreg(np)

    def do_target(self, np):
        np = self._(np)
        if len(np) < 1:
            print("Please enter target column")
        self.data._target(np[0])
        self.data._drop(np[0])


if __name__ == '__main__':
    dt = Shell(sys.argv[1])
    dt.cmdloop()
