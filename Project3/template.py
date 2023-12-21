from typing import Dict, Tuple, List
import numpy as np
import torch.nn as nn
from torch.nn import Embedding
import torch.optim as optim
import numpy as np
import torch


class KGRS:

    def __init__(self, config: Dict, train_pos: np.array, train_neg: np.array, kg_lines: List[str]):
        """
        :param config: The HyperParameter Dict of your Algorithm
        :param train_pos: The Positive Samples in the Training Set, is a numpy matrix
            with shape (n,3), while `n` is the number of positive
            samples,
            and in each sample, the first number represent the user, the
            second represent the item, and the last indicate interest or
            not. e.g. [[1,2,1], [2,5,1],[1,3,1]] indicate that user 1 has
            interest in item 2 and item 3, user 2 has interest in item 5.

        :param train_neg: The Negative Samples in the Training Set, is a numpy matrix
            with shape (n,3), while `n` is the number of positive
            samples, and in each sample, the first number represent the
            user, the second represent the item, and the last indicate
            interest or not. e.g. [[1,4,0], [2,2,0],[1,5,0]] indicate
            that user 1 has no interest in item 4 and item 5, user 2 has
            no interest in item 2.

        :param kg_lines: The Knowledge Graph Lines, is a list of strings. Each
            element in the list is a string representing one relation
            in the Knowledge Graph. The string can be split into 3
            parts by '\t', the first part is head entity, the second
            part is relation type, and the third part is tail entity.
            E.g. ["749\tfilm.film.writer\t2347"] represent a Knowledge
            Graph only has one relation, in that relation, head entity
            is 749, tail entity is 2347, and the relation type is
            "film.film.writer".
            """


    def predict_score(self, user_id, item_id):
        '''
        predict the score for user_id to item_id. The higher means the user is more interested in the item
        '''
        


    def eval_ctr(self, test_data: np.array) -> np.array:
        """
        Evaluate the CTR Task result
        :param test_data: The test data that you need to predict. The data is a numpy
            matrix with shape (n, 2), while `n` is the number of the test
            samples, and in each sample. 
            The first dimension is the user and the second is the item. 
            e.g. [[2, 4], [2, 6], [4, 1]]
            means you need to predict the interest level of: from user 2
            to item 4, from user 2 to item 6, and from user 4 to item 1.

        :return: The prediction result, is an n dimension numpy array, and the i-th
            dimension means the predicted interest level of the i-th sample,
            while the higher score means user has higher interest in the item.
            e.g. while test_data=[[2, 4], [2, 6], [4, 1]], the return value [1.2,
            3.3, 0.7] means that the interest level from user 2 to item 6 is
            highest in these samples, and interest level from user 2 to item 4 is
            second highest, interest level from user 4 to item 1 is lowest.
        """


    def eval_topk(self, users: List[int], k: int = 5) -> List[List[int]]:
        """
        Evaluate the Top-K Recommendation Task result
        :param users: The list of the id of the users that need to be recommended
            items. e.g. [2, 4, 8] means you need to recommend k items for
            the user 2, 4, 8 respectively, and the term of the user and
            recommended item cannot have appeared in the train_pos data.

        :param k: The number of the items recommended to each user. In this project, k=5.

        :return: The items recommended to the users respectively, and the order of the
                items should be sorted by the interest level of the user to the item.
                e.g. while user=[2, 4, 8] and k=5, the return value is [[2, 5, 7, 4,
                6],[3, 5, 2, 1, 21],[12, 43, 7, 3, 2]] means you will recommend item
                2, 5, 7, 4, 6 to user 2, recommend item 3, 5, 2, 1, 21 to user 4, and
                recommend item 12, 43, 7, 3, 2 to user 8, and the interest level from
                user to the item in the recommend list are degressive.
        """
        
