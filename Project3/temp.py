from typing import Dict, Tuple, List
import numpy as np
import torch.nn as nn
from torch.nn import Embedding
import torch.optim as optim
import numpy as np
import torch


class KnowledgeGraphEmbedding(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(KnowledgeGraphEmbedding, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

    def forward(self, head, relation, tail):
        head_embedding = self.entity_embeddings(head)
        relation_embedding = self.relation_embeddings(relation)
        tail_embedding = self.entity_embeddings(tail)

        return head_embedding, relation_embedding, tail_embedding


class KGRS:

    def __init__(self, config: Dict, train_pos: np.array, train_neg: np.array, kg_lines: List[str]):
        self.config = config
        self.train_pos = train_pos
        self.train_neg = train_neg
        self.kg_lines = kg_lines
        self.device = torch.device('cpu')
        self.max_item_id = max(self.train_neg[:, 1])
        self.item_num = self.max_item_id+1

        # KG information processing
        self.relation_to_id = {}  # Mapping relations to unique IDs
        self.id_to_relation = {}  # Mapping unique IDs to relations
        self.create_mappings()  # Function to create mappings from KG

        self.entity_embeddings = Embedding(
            self.item_num, self.config['embedding_dim'])
        self.relation_embeddings = Embedding(
            len(self.relation_to_id), self.config['embedding_dim'])

        self.initialize_embeddings()

        self.kg_embedding_model = KnowledgeGraphEmbedding(
            num_entities=self.item_num,
            num_relations=len(self.relation_to_id),
            embedding_dim=self.config['embedding_dim']
        )
        self.optimizer_kg = optim.Adam(
            self.kg_embedding_model.parameters(), lr=self.config['learning_rate'])

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
        entity_embeddings = np.random.randn(
            self.item_num, self.config['embedding_dim'])
        relation_embeddings = np.random.randn(
            len(self.relation_to_id), self.config['embedding_dim'])

        self.entity_embeddings.weight.data.copy_(
            torch.from_numpy(entity_embeddings))
        self.relation_embeddings.weight.data.copy_(
            torch.from_numpy(relation_embeddings))

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

    def train_kg_embeddings(self, num_epochs=20):
        # 为知识图谱训练嵌入向量
        for epoch in range(num_epochs):
            for line in self.kg_lines:
                head, relation, tail = map(int, line.strip().split('\t'))
                # convert to torch tensors
                head = torch.LongTensor([head])
                relation = torch.LongTensor([self.relation_to_id[relation]])
                tail = torch.LongTensor([tail])

                self.optimizer_kg.zero_grad()
                head_emb, rel_emb, tail_emb = self.kg_embedding_model(
                    head, relation, tail)
                loss = self.calculate_kg_loss(head_emb, rel_emb, tail_emb)
                loss.backward()
                self.optimizer_kg.step()

                # Print training statistics
                if epoch % 10 == 0:
                    print(
                        f'Epoch [{epoch}/{num_epochs}], KG Loss: {loss.item()}')

    def calculate_kg_loss(self, head_emb, rel_emb, tail_emb):
        # loss function
        margin = self.config['margin']
        loss = torch.mean((head_emb + rel_emb - tail_emb) ** 2)  # 例如采用平方损失函数
        return loss

    def predict_interest_with_kg(self, user_id, item_id):
        # 利用知识图谱信息预测用户对物品的兴趣程度
        user_tensor = torch.LongTensor([user_id]).to(self.device)
        item_tensor = torch.LongTensor([item_id]).to(self.device)

        # 获取用户和物品的基本嵌入向量
        user_embedding = self.model.get_user_embedding(user_tensor)
        item_embedding = self.model.get_item_embedding(item_tensor)

        # 假设你已经能够获取知识图谱中的嵌入向量，以下是示例代码，获取嵌入向量的方法可能有所不同
        with torch.no_grad():
            head_emb, rel_emb, tail_emb = self.kg_embedding_model(
                item_tensor, torch.LongTensor([0]), item_tensor)

        # 例如，将用户嵌入向量、物品嵌入向量以及知识图谱中的嵌入向量进行拼接或加权平均
        combined_embedding = torch.cat(
            [user_embedding, item_embedding, head_emb, rel_emb, tail_emb], dim=-1)
        # 或者使用其他结合方式，如加权求和、点乘等

        predicted_score = self.model.predict_interest(combined_embedding)

        return predicted_score



    def eval_ctr(self, test_data: np.array) -> np.array:
        """
            The first dimension is the user and the second is the item. 
        """
        predictions = []
        for sample in test_data:
            user_id, item_id = sample[0], sample[1]
            predicted_score = self.predict_score(user_id, item_id)
            predictions.append(predicted_score)

        return np.array(predictions)

    def eval_topk(self, users: List[int], k: int = 5) -> List[List[int]]:
        recommendations = []
        for user_id in users:
            user_recommendations = []
            # Items already interacted by the user
            user_item_ids = set(
                self.train_pos[self.train_pos[:, 0] == user_id][:, 1])

            item_scores = []
            for item_id in range(0, self.max_item_id + 1):
                if item_id not in user_item_ids:  # Avoid recommending items the user has interacted with
                    score = self.predict_score(user_id, item_id)
                    item_scores.append((item_id, score))

            item_scores.sort(key=lambda x: x[1], reverse=True)  # sort by score

            # top-k
            top_k_items = [item_id for item_id, _ in item_scores[:k]]
            recommendations.append(top_k_items)

        return recommendations
