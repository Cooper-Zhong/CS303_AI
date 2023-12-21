import random
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from typing import Dict, Tuple, List


class KGRS:

    def __init__(self, train_pos: np.array, train_neg: np.array, kg_lines: List[str]):
        self.train_pos = train_pos
        self.train_neg = train_neg
        self.num_users = np.max(self.train_pos[:, 0]) + 1
        self.num_items = np.max(self.train_pos[:, 1]) + 1

        self.kg_lines = kg_lines
        # self.kg_relation_dict = self.encode_kg()
        # self.kg_matrix = self.get_kg_matrix()
        config = {
            'num_factors': 20,
            'learning_rate': 0.01,
            'regularization': 0.002,
            'epochs': 60
        }
        self.num_factors = config['num_factors']  # Number of latent factors
        self.lr = config['learning_rate']  # Learning rate
        self.reg = config['regularization']  # Regularization strength
        self.num_epochs = config['epochs']  # Number of training epochs

        # Initialize user and item matrices with random values
        self.user_matrix = np.random.normal(
            scale=1.0/self.num_factors, size=(self.num_users, self.num_factors))
        self.item_matrix = np.random.normal(
            scale=1.0/self.num_factors, size=(self.num_items, self.num_factors))

    def get_kg_matrix(self):
        kg_matrix = np.zeros((self.num_items, self.num_items))
        for line in self.kg_lines:
            parts = line.split('\t')
            item_id = int(parts[0])
            related_item_id = int(parts[2])
            kg_matrix[item_id, related_item_id] = 1
        return kg_matrix

    # def encode_kg(self):
    #     # Encode kg_matrix using one-hot encoding
    #     relations = set()
    #     for line in self.kg_lines:
    #         parts = line.split('\t')
    #         relations.add(parts[1])
    #     relations = list(relations)
    #     encoder = OneHotEncoder(sparse=False)

    #     # 将关系列表转换为二维数组形式（每个关系为一个列表）
    #     relations_array = [[relation] for relation in relations]

    #     # 将关系列表进行独热编码
    #     encoded_relations = encoder.fit_transform(relations_array)

    #     print("独热编码结果：")
    #     print(encoded_relations)
    #     kg_relation_dict = {relation: encoded_relations[i]
    #                         for i, relation in enumerate(relations)}
    #     return kg_relation_dict

    def training(self):
        for epoch in range(self.num_epochs):
            # Shuffle training data
            np.random.shuffle(self.train_pos)
            np.random.shuffle(self.train_neg)

            for sample in np.concatenate((self.train_pos, self.train_neg)):
                user_id, item_id, label = sample

                # Compute prediction
                prediction = np.dot(
                    self.user_matrix[user_id], self.item_matrix[item_id])
                error = label - prediction

                # Update user and item matrices using gradient descent
                self.user_matrix[user_id] += self.lr * (
                    error * self.item_matrix[item_id] - self.reg * self.user_matrix[user_id])
                self.item_matrix[item_id] += self.lr * (
                    error * self.user_matrix[user_id] - self.reg * self.item_matrix[item_id])

            # Evaluate model
            if epoch % 10 == 0:
                print('Epoch: %d' % epoch)

    def predict_score(self, user_id, item_id):
        # Predict the score for user_id to item_id
        return np.dot(self.user_matrix[user_id], self.item_matrix[item_id])

    def eval_ctr(self, test_data: np.array) -> np.array:
        predictions = []
        for user_id, item_id in test_data:
            predicted_score = self.predict_score(user_id, item_id)
            predictions.append(predicted_score)

        return np.array(predictions)

    def eval_topk(self, users: List[int], k: int = 5) -> List[List[int]]:
        recommendations = []
        for user_id in users:
            user_recommendations = []
            train_interacted_items = set(
                self.train_pos[self.train_pos[:, 0] == user_id][:, 1])
            neg_items = set(
                self.train_neg[self.train_neg[:, 0] == user_id][:, 1])
            for item_id in range(self.num_items):
                # do not recommend items that are already interacted in training set
                if item_id not in train_interacted_items:
                    score = self.predict_score(user_id, item_id)
                    user_recommendations.append((item_id, score))

            # Sort recommendations by predicted score and select top-k
            user_recommendations.sort(key=lambda x: x[1], reverse=True)
            final_user_recommendations = []
            cnt = 0
            for rec_item in user_recommendations[:2*k]:
                if cnt >= k:
                    break
                final_user_recommendations.append(rec_item[0])

            recommendations.append(final_user_recommendations)
        return recommendations
