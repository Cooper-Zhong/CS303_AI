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
            """
        self.config = config
        self.train_pos = train_pos
        self.train_neg = train_neg
        self.kg_lines = kg_lines
        self.device = torch.device('cpu')
        self.max_item_id = max(self.train_neg[:, 1])
        self.item_num = self.max_item_id+1

        # KG information processing
        self.entity_to_id = {}  # Mapping entities to unique IDs
        self.relation_to_id = {}  # Mapping relations to unique IDs
        self.id_to_entity = {}  # Mapping unique IDs to entities
        self.id_to_relation = {}  # Mapping unique IDs to relations
        self.create_mappings()  # Function to create mappings from KG

        self.entity_embeddings = Embedding(
            self.item_num, self.config['embedding_dim'])
        self.relation_embeddings = Embedding(
            len(self.relation_to_id), self.config['embedding_dim'])
        
        self.initialize_embeddings() 

    def create_mappings(self):
        # Process kg_lines and create mappings
        entity_id = 0
        relation_id = 0
        for line in self.kg_lines:
            head, relation, tail = line.strip().split('\t')
            if relation not in self.relation_to_id:
                self.relation_to_id[relation] = relation_id
                self.id_to_relation[relation_id] = relation
                relation_id += 1

    def initialize_embeddings(self):
        # Initialize entity and relation embeddings
        entity_embeddings = np.random.randn(self.item_num, self.config['embedding_dim'])
        relation_embeddings = np.random.randn(len(self.relation_to_id), self.config['embedding_dim'])
        
        self.entity_embeddings.weight.data.copy_(torch.from_numpy(entity_embeddings))
        self.relation_embeddings.weight.data.copy_(torch.from_numpy(relation_embeddings))
   

    def initialize_model(self):
        # Define the neural network architecture
        input_dim = 2  # (user_id, item_id)
        hidden_dim = 64
        output_dim = 1  # score
        model = nn.Sequential(
            nn.Embedding(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # Sigmoid activation for score prediction
        )
        return model.to(self.device)


    def predict_score(self, user_id, item_id):
        # Convert user_id and item_id to torch tensors
        user_tensor = torch.LongTensor([user_id]).to(self.device)
        item_tensor = torch.LongTensor([item_id]).to(self.device)

        with torch.no_grad():
            self.model.eval()  # Set the model to evaluation mode
            predicted_score = self.model(user_tensor, item_tensor)

        # Return the predicted score as a Python float
        return predicted_score.item()


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
        predictions = []

        for sample in test_data:
            user_id, item_id = sample[0], sample[1]
            # Assuming you have a function `predict_score` that predicts the interest level
            # You'll need to replace this with your actual prediction function
            predicted_score = self.predict(user_id, item_id)
            predictions.append(predicted_score)

        return np.array(predictions)

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
        recommendations = []

        for user_id in users:
            user_recommendations = []
            # Items already interacted by the user
            user_item_ids = set(
                self.train_pos[self.train_pos[:, 0] == user_id][:, 1])

            # Predict scores for all items for the user
            item_scores = []
            # Assuming max_item_id is the maximum item index
            for item_id in range(0, self.max_item_id + 1):
                if item_id not in user_item_ids:  # Avoid recommending items the user has interacted with
                    score = self.predict_score(user_id, item_id)
                    item_scores.append((item_id, score))

            item_scores.sort(key=lambda x: x[1], reverse=True)  # sort by score

            # top-k
            top_k_items = [item_id for item_id, _ in item_scores[:k]]
            recommendations.append(top_k_items)

        return recommendations
