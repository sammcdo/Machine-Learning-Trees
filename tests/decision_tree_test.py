import unittest
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from decision_tree import DecisionTree

class DecisionTreeTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(DecisionTreeTest, self).__init__(*args, **kwargs)
        self.dt = DecisionTree()

    def test_gini_worstCase(self):
        testGroups = [[[1, 1], [1, 0]], [[1, 1], [1, 0]]]
        testClasses = [0,1]
        self.assertEqual(self.dt.gini_index(testGroups, testClasses), 0.5)
    
    def test_gini_bestCase(self):
        testGroups = [[[1, 0], [1, 0]], [[1, 1], [1, 1]]]
        testClasses = [0,1]
        self.assertEqual(self.dt.gini_index(testGroups, testClasses), 0)
    

if __name__ == "__main__":
    unittest.main()