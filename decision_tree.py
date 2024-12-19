import pandas as pd
import random

class DecisionTree:
    def __init__(self, maxDepth=5, minSize=10):
        self.max_depth = maxDepth
        self.min_size = minSize

    def fit(self, Xs: pd.DataFrame, ys: pd.DataFrame):
        """
        Assumes the ys are a one column pandas dataframe
        """
        self.Xs = Xs
        self.ys = ys
        self.Xs['$target$'] = self.ys
        self.classes = list(set(self.ys.iloc[:, 0]))
        self.tree = self._buildTree(Xs)
    
    def _getSplit(self, data):
        b_index = -1
        b_value = b_score = float('inf')
        b_groups = None
        for c in data.columns:
            if c == "$target$":
                continue
            for i, r in data.iterrows():
                groups = self._testSplit(c, r[c], data)
                gini = self._giniIndex(groups, self.classes)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = c, r[c], gini, groups
        return {'index': b_index, 'value': b_value, 'groups': b_groups}
    
    def _testSplit(self, col, value, data):
        """
        Split the dataset based on split value for column
        If it is less than the value the row goes in left, otherwise
        it goes in right.
        """
        left = data[data[col] < value]
        right = data[data[col] >= value]
        return left, right

    def _giniIndex(self, groups, classes):
        """
        Given a selection of data points, determine how mixed they are 
        into different classes. A score of 0 means all belong to the same 
        class and 0.5 is the worst case meaning all classes have even 
        representation
        """
        n_instances = float(sum([len(group) for group in groups]))
        gini = 0.0
        for group in groups:
            size = float(len(group))
            if size == 0:
                continue
            score = 0.0
            for class_val in classes:
                p = (group["$target$"] == class_val).sum() / size
                score += p * p
            gini += (1.0 - score) * (size / n_instances)
        return gini
    
    def _toTerminalNode(self, group):
        outcomes = [row["$target$"] for _, row in group.iterrows()]
        return max(set(outcomes), key=outcomes.count)
    
    def _split(self, node, max_depth, min_size, depth):
        left, right = node['groups']
        del (node['groups'])

        if len(left) == 0 or len(right) == 0:
            node['left'] = node['right'] = self._toTerminalNode(pd.concat([left, right]))
            return
        if depth >= max_depth:
            node['left'], node['right'] = self._toTerminalNode(left), self._toTerminalNode(right)
            return
        
        if len(left) <= min_size:
            node['left'] = self._toTerminalNode(left)
        else:
            node['left'] = self._getSplit(left)
            self._split(node['left'], max_depth, min_size, depth + 1)
        
        if len(right) <= min_size:
            node['right'] = self._toTerminalNode(right)
        else:
            node['right'] = self._getSplit(right)
            self._split(node['right'], max_depth, min_size, depth + 1)
        
    def _buildTree(self, train):
        root = self._getSplit(train)
        self._split(root, self.max_depth, self.min_size, 1)
        self.root = root
    
    def printTree(self):
        self._printTree(self.root)
    
    def _printTree(self, node, depth=0):
        if isinstance(node, dict):
            print('%s[X%d < %.3f]' % ((depth * ' ', node['index'], node['value'])))
            self._printTree(node['left'], depth + 1)
            self._printTree(node['right'], depth + 1)
        else:
            print('%s[%s]' % ((depth * ' ', node)))
    
    def _predict(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self._predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self._predict(node['right'], row)
            else:
                return node['right']
    
    def predict(self, Xs: pd.DataFrame):
        return [self._predict(self.root, r) for _, r in Xs.iterrows()]

if __name__ == "__main__":
    random.seed(1)
    df = pd.read_csv("./datasets/data_banknote_authentication.csv", header=None)
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    Xs = df.iloc[:, :-1]
    ys = df.iloc[:, -1].to_frame()

    splitInd = int(len(Xs) * 0.8)

    trainX = Xs.iloc[0:splitInd]
    testX  = Xs.iloc[splitInd:]
    trainY = ys.iloc[0:splitInd]
    testY  = ys.iloc[splitInd:]


    dt = DecisionTree()

    print(df.columns)
    print(testY, type(testY))

    dt.fit(trainX, trainY)
    p = dt.predict(testX)
    print(p)
    print(testY)

    c = 0
    testY.reset_index(drop=True, inplace=True)
    for i, r in testY.iterrows():
        if p[i] == testY.iat[i, 0]:
            c += 1
    
    print("Accuracy:", c / len(p))

    dt.printTree()
