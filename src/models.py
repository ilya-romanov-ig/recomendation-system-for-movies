import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class UBCF:
    def __init__(self, metric='cosine', k_similar=50, min_similar_users=1):
        self.metric = metric
        self.k_similar = k_similar
        self.min_similar_users = min_similar_users
        self.user_item_matrix = None
        self.user_similarities = None
        self.n_users = None
        self.n_items = None
        
    def fit(self, user_item_matrix):
        self.user_item_matrix = user_item_matrix
        self.n_users, self.n_items = user_item_matrix.shape
        
        if self.metric == 'cosine':
            self.user_similarities = cosine_similarity(user_item_matrix)
        elif self.metric == 'pearson':
            dense_matrix = user_item_matrix.toarray()
            dense_matrix = dense_matrix.astype(float)
            dense_matrix[dense_matrix == 0] = np.nan
            self.user_similarities = np.corrcoef(dense_matrix, rowvar=True)
            self.user_similarities = np.nan_to_num(self.user_similarities)
        
        np.fill_diagonal(self.user_similarities, 0)
        return self
    
    def _predict_rating(self, user_id, item_id):
        if user_id >= self.n_users:
            return 0
        
        item_ratings = self.user_item_matrix[:, item_id].toarray().flatten()
        rated_user_indices = np.where(item_ratings > 0)[0]
        
        if len(rated_user_indices) == 0:
            return 0
        
        sims = self.user_similarities[user_id, rated_user_indices]
        ratings = item_ratings[rated_user_indices]
        
        positive_mask = sims > 0
        sims_positive = sims[positive_mask]
        ratings_positive = ratings[positive_mask]
        
        if len(sims_positive) < self.min_similar_users:
            return 0
        
        sorted_indices = np.argsort(sims_positive)[::-1]
        
        top_k_indices = sorted_indices[:self.k_similar]
        top_k_sims = sims_positive[top_k_indices]
        top_k_ratings = ratings_positive[top_k_indices]
        
        pred_rating = np.dot(top_k_sims, top_k_ratings) / np.sum(top_k_sims)
        
        return pred_rating
    
    def recommend(self, user_id, n_recommendations=10):
        if user_id >= self.n_users:
            return []
        
        user_ratings = self.user_item_matrix[user_id, :].toarray().flatten()
        
        unrated_item_indices = np.where(user_ratings == 0)[0]
        
        preds = []
        for item_id in unrated_item_indices:
            pred = self._predict_rating(user_id, item_id)
            if pred > 0:  
                preds.append((item_id, pred))
        
        preds.sort(key=lambda x: x[1], reverse=True)
        
        return preds[:n_recommendations]

class IBCF:
    def __init__(self, metric='cosine', k_similar=50, min_similar_items=1):
        self.metric = metric
        self.k_similar = k_similar
        self.min_similar_items = min_similar_items
        self.user_item_matrix = None
        self.item_similarities = None
        self.n_users = None
        self.n_items = None
        
    def fit(self, user_item_matrix):
        self.user_item_matrix = user_item_matrix
        self.n_users, self.n_items = user_item_matrix.shape
        
        if self.metric == 'cosine':
            self.item_similarities = cosine_similarity(user_item_matrix.T)
        elif self.metric == 'adjusted_cosine':
            dense_matrix = user_item_matrix.toarray().astype(float)
            
            user_means = np.zeros(self.n_users)
            for i in range(self.n_users):
                user_ratings = dense_matrix[i, :]
                rated_mask = user_ratings > 0
                if np.sum(rated_mask) > 0:
                    user_means[i] = np.mean(user_ratings[rated_mask])
            
            matrix_norm = dense_matrix.copy()
            for i in range(self.n_users):
                rated_mask = dense_matrix[i, :] > 0
                matrix_norm[i, rated_mask] -= user_means[i]
            
            matrix_norm[dense_matrix == 0] = 0
            self.item_similarities = cosine_similarity(matrix_norm.T)
        
        np.fill_diagonal(self.item_similarities, 0)

        return self
    
    def _predict_rating(self, user_id, item_id):
        if user_id >= self.n_users or item_id >= self.n_items:
            return 0
        
        user_ratings = self.user_item_matrix[user_id, :].toarray().flatten()
        
        rated_item_indices = np.where(user_ratings > 0)[0]
        
        if len(rated_item_indices) == 0:
            return 0
        
        similarities = self.item_similarities[item_id, rated_item_indices]
        user_rated_ratings = user_ratings[rated_item_indices]
        
        positive_mask = similarities > 0
        positive_similarities = similarities[positive_mask]
        positive_ratings = user_rated_ratings[positive_mask]
        
        if len(positive_similarities) < self.min_similar_items:
            return 0
        
        sorted_indices = np.argsort(positive_similarities)[::-1]
        top_k_indices = sorted_indices[:self.k_similar]
        
        top_k_sims = positive_similarities[top_k_indices]
        top_k_ratings = positive_ratings[top_k_indices]
        
        if np.sum(top_k_sims) == 0:
            return 0
            
        pred_rating = np.dot(top_k_sims, top_k_ratings) / np.sum(top_k_sims)
        return pred_rating
    
    def recommend(self, user_id, n_recommendations=10):
        if user_id >= self.n_users:
            return []
        
        user_ratings = self.user_item_matrix[user_id, :].toarray().flatten()
        
        unrated_item_indices = np.where(user_ratings == 0)[0]
        
        preds = []
        for item_id in unrated_item_indices:
            pred = self._predict_rating(user_id, item_id)
            if pred > 0:
                preds.append((item_id, pred))
        
        preds.sort(key=lambda x: x[1], reverse=True)
        
        return preds[:n_recommendations]