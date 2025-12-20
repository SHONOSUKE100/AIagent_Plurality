"""Content moderation and recommendation algorithms for the simulation.

This module provides various recommendation algorithms that can be used
to control how content is distributed to agents in the simulation.
These algorithms aim to study the effects of different content distribution
strategies on social dynamics, including echo chamber formation and polarization.

Available Algorithms:
    - RandomRecommender: Random baseline
    - CollaborativeFilteringRecommender: User-based collaborative filtering
    - BridgingRecommender: Promotes content that bridges opinion clusters
    - DiversityRecommender: Maximizes content diversity
    - EchoChambberRecommender: Intentionally creates echo chambers (for study)
"""

from __future__ import annotations

import random
from pydantic import BaseModel, Field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class RecommendationType(str, Enum):
    """Types of recommendation algorithms available."""
    
    RANDOM = "random"
    COLLABORATIVE = "collaborative"
    BRIDGING = "bridging"
    DIVERSITY = "diversity"
    ECHO_CHAMBER = "echo_chamber"
    HYBRID = "hybrid"


class Post(BaseModel):
    """Represents a post in the simulation."""
    
    post_id: str
    user_id: int
    content: str
    created_at: str
    num_likes: int = 0
    num_dislikes: int = 0
    embedding: Optional[np.ndarray] = None
    topics: List[str] = Field(default_factory=list)
    sentiment_score: float = 0.0  # -1.0 (negative) to 1.0 (positive)


class User(BaseModel):
    """Represents a user/agent in the simulation."""
    
    user_id: int
    bio: str = ""
    embedding: Optional[np.ndarray] = None
    interests: List[str] = Field(default_factory=list)
    opinion_vector: Optional[np.ndarray] = None  # Opinion on various topics
    num_followers: int = 0


class Interaction(BaseModel):
    """Represents a user interaction with a post."""
    
    user_id: int
    post_id: str
    action: str  # 'like', 'dislike', 'comment', 'repost', 'view'
    timestamp: str


class BaseRecommender(ABC):
    """Abstract base class for recommendation algorithms."""
    
    def __init__(self, max_recommendations: int = 10):
        """Initialize the recommender.
        
        Args:
            max_recommendations: Maximum number of posts to recommend per user.
        """
        self.max_recommendations = max_recommendations
    
    @abstractmethod
    def recommend(
        self,
        user: User,
        posts: Sequence[Post],
        interactions: Sequence[Interaction],
        all_users: Sequence[User],
    ) -> List[str]:
        """Generate recommendations for a user.
        
        Args:
            user: The target user to recommend posts to.
            posts: All available posts.
            interactions: All user interactions.
            all_users: All users in the system.
            
        Returns:
            List of post IDs to recommend.
        """
        pass
    
    def recommend_batch(
        self,
        users: Sequence[User],
        posts: Sequence[Post],
        interactions: Sequence[Interaction],
    ) -> Dict[int, List[str]]:
        """Generate recommendations for multiple users.
        
        Args:
            users: List of users to recommend to.
            posts: All available posts.
            interactions: All user interactions.
            
        Returns:
            Dictionary mapping user_id to list of recommended post IDs.
        """
        return {
            user.user_id: self.recommend(user, posts, interactions, users)
            for user in users
        }
    
    def _filter_own_posts(self, user: User, posts: Sequence[Post]) -> List[Post]:
        """Filter out posts created by the user themselves."""
        return [p for p in posts if p.user_id != user.user_id]
    
    def _compute_cosine_similarity(
        self, 
        vec1: np.ndarray, 
        vec2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two vectors."""
        if vec1 is None or vec2 is None:
            return 0.0
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))


class RandomRecommender(BaseRecommender):
    """Random recommendation baseline.
    
    This recommender randomly samples posts for each user.
    Useful as a baseline for comparison with other algorithms.
    """
    
    def recommend(
        self,
        user: User,
        posts: Sequence[Post],
        interactions: Sequence[Interaction],
        all_users: Sequence[User],
    ) -> List[str]:
        available_posts = self._filter_own_posts(user, posts)
        if not available_posts:
            return []
        
        sample_size = min(self.max_recommendations, len(available_posts))
        selected = random.sample(available_posts, sample_size)
        return [p.post_id for p in selected]


class CollaborativeFilteringRecommender(BaseRecommender):
    """User-based collaborative filtering recommender.
    
    Recommends posts that similar users have liked.
    This algorithm may reinforce echo chambers by recommending
    content that users with similar opinions have engaged with.
    """
    
    def __init__(
        self, 
        max_recommendations: int = 10,
        similarity_threshold: float = 0.5,
        k_neighbors: int = 5,
    ):
        super().__init__(max_recommendations)
        self.similarity_threshold = similarity_threshold
        self.k_neighbors = k_neighbors
    
    def recommend(
        self,
        user: User,
        posts: Sequence[Post],
        interactions: Sequence[Interaction],
        all_users: Sequence[User],
    ) -> List[str]:
        available_posts = self._filter_own_posts(user, posts)
        if not available_posts:
            return []
        
        # Find similar users based on embedding or opinion vector
        similar_users = self._find_similar_users(user, all_users)
        
        # Get posts liked by similar users
        user_interactions = {i.user_id: [] for i in interactions}
        for interaction in interactions:
            if interaction.action in ('like', 'repost'):
                user_interactions[interaction.user_id].append(interaction.post_id)
        
        # Score posts based on similar users' interactions
        post_scores: Dict[str, float] = {}
        for similar_user, similarity in similar_users:
            for post_id in user_interactions.get(similar_user.user_id, []):
                if post_id not in post_scores:
                    post_scores[post_id] = 0.0
                post_scores[post_id] += similarity
        
        # If no collaborative data, fall back to random
        if not post_scores:
            sample_size = min(self.max_recommendations, len(available_posts))
            selected = random.sample(available_posts, sample_size)
            return [p.post_id for p in selected]
        
        # Sort and return top recommendations
        available_post_ids = {p.post_id for p in available_posts}
        scored_posts = [
            (pid, score) for pid, score in post_scores.items()
            if pid in available_post_ids
        ]
        scored_posts.sort(key=lambda x: x[1], reverse=True)
        
        return [pid for pid, _ in scored_posts[:self.max_recommendations]]
    
    def _find_similar_users(
        self, 
        user: User, 
        all_users: Sequence[User]
    ) -> List[Tuple[User, float]]:
        """Find users similar to the target user."""
        similarities = []
        
        for other_user in all_users:
            if other_user.user_id == user.user_id:
                continue
            
            # Compute similarity based on embeddings or opinion vectors
            if user.embedding is not None and other_user.embedding is not None:
                sim = self._compute_cosine_similarity(user.embedding, other_user.embedding)
            elif user.opinion_vector is not None and other_user.opinion_vector is not None:
                sim = self._compute_cosine_similarity(user.opinion_vector, other_user.opinion_vector)
            else:
                sim = 0.0
            
            if sim >= self.similarity_threshold:
                similarities.append((other_user, sim))
        
        # Return top k neighbors
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:self.k_neighbors]


class BridgingRecommender(BaseRecommender):
    """Bridging-based recommender that promotes cross-cluster content.
    
    This algorithm prioritizes content that bridges different opinion clusters,
    aiming to reduce polarization by exposing users to diverse perspectives
    that are still somewhat related to their interests.
    """
    
    def __init__(
        self,
        max_recommendations: int = 10,
        bridging_weight: float = 0.7,
        relevance_weight: float = 0.3,
    ):
        super().__init__(max_recommendations)
        self.bridging_weight = bridging_weight
        self.relevance_weight = relevance_weight
    
    def recommend(
        self,
        user: User,
        posts: Sequence[Post],
        interactions: Sequence[Interaction],
        all_users: Sequence[User],
    ) -> List[str]:
        available_posts = self._filter_own_posts(user, posts)
        if not available_posts:
            return []
        
        # Score posts based on bridging potential and relevance
        post_scores = []
        for post in available_posts:
            bridging_score = self._compute_bridging_score(post, user, all_users, interactions)
            relevance_score = self._compute_relevance_score(post, user)
            
            combined_score = (
                self.bridging_weight * bridging_score +
                self.relevance_weight * relevance_score
            )
            post_scores.append((post.post_id, combined_score))
        
        # Sort by combined score
        post_scores.sort(key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in post_scores[:self.max_recommendations]]
    
    def _compute_bridging_score(
        self,
        post: Post,
        user: User,
        all_users: Sequence[User],
        interactions: Sequence[Interaction],
    ) -> float:
        """Compute how well a post bridges different opinion clusters.
        
        A good bridging post is one that has been liked/engaged by users
        from different opinion clusters.
        """
        # Get users who interacted positively with this post
        engaging_user_ids = {
            i.user_id for i in interactions
            if i.post_id == post.post_id and i.action in ('like', 'repost', 'comment')
        }
        
        if len(engaging_user_ids) < 2:
            return 0.0
        
        # Compute opinion diversity among engaging users
        engaging_users = [u for u in all_users if u.user_id in engaging_user_ids]
        if not engaging_users or all(u.opinion_vector is None for u in engaging_users):
            return random.random() * 0.5  # Low random score if no opinion data
        
        # Calculate variance in opinion vectors as a measure of diversity
        valid_opinions = [u.opinion_vector for u in engaging_users if u.opinion_vector is not None]
        if len(valid_opinions) < 2:
            return 0.0
        
        opinion_matrix = np.array(valid_opinions)
        variance = np.mean(np.var(opinion_matrix, axis=0))
        
        # Normalize to 0-1 range (assuming variance typically < 1)
        return min(1.0, variance * 2)
    
    def _compute_relevance_score(self, post: Post, user: User) -> float:
        """Compute relevance of post to user's interests."""
        if post.embedding is not None and user.embedding is not None:
            return (self._compute_cosine_similarity(post.embedding, user.embedding) + 1) / 2
        return 0.5  # Neutral score if no embeddings


class DiversityRecommender(BaseRecommender):
    """Diversity-maximizing recommender.
    
    This algorithm aims to maximize the diversity of recommended content
    by selecting posts that cover different topics and viewpoints.
    """
    
    def __init__(
        self,
        max_recommendations: int = 10,
        diversity_weight: float = 0.6,
        recency_weight: float = 0.2,
        popularity_weight: float = 0.2,
    ):
        super().__init__(max_recommendations)
        self.diversity_weight = diversity_weight
        self.recency_weight = recency_weight
        self.popularity_weight = popularity_weight
    
    def recommend(
        self,
        user: User,
        posts: Sequence[Post],
        interactions: Sequence[Interaction],
        all_users: Sequence[User],
    ) -> List[str]:
        available_posts = self._filter_own_posts(user, posts)
        if not available_posts:
            return []
        
        # Use greedy selection to maximize diversity
        selected: List[Post] = []
        remaining = list(available_posts)
        
        while len(selected) < self.max_recommendations and remaining:
            best_post = None
            best_score = -float('inf')
            
            for post in remaining:
                score = self._compute_marginal_gain(post, selected, remaining)
                if score > best_score:
                    best_score = score
                    best_post = post
            
            if best_post:
                selected.append(best_post)
                remaining.remove(best_post)
            else:
                break
        
        return [p.post_id for p in selected]
    
    def _compute_marginal_gain(
        self,
        post: Post,
        selected: List[Post],
        all_posts: List[Post],
    ) -> float:
        """Compute the marginal gain of adding a post to the selection."""
        # Diversity: how different is this post from already selected posts
        if selected:
            similarities = []
            for sel_post in selected:
                if post.embedding is not None and sel_post.embedding is not None:
                    sim = self._compute_cosine_similarity(post.embedding, sel_post.embedding)
                    similarities.append(sim)
            
            if similarities:
                diversity_score = 1 - max(similarities)  # Lower similarity = higher diversity
            else:
                diversity_score = 0.5
        else:
            diversity_score = 1.0  # First post gets full diversity score
        
        # Recency score (normalized by position in list)
        try:
            recency_idx = all_posts.index(post)
            recency_score = 1 - (recency_idx / len(all_posts))
        except (ValueError, ZeroDivisionError):
            recency_score = 0.5
        
        # Popularity score
        total_engagement = post.num_likes + abs(post.num_dislikes)
        max_engagement = max(
            (p.num_likes + abs(p.num_dislikes) for p in all_posts),
            default=1
        )
        popularity_score = total_engagement / max_engagement if max_engagement > 0 else 0
        
        return (
            self.diversity_weight * diversity_score +
            self.recency_weight * recency_score +
            self.popularity_weight * popularity_score
        )


class EchoChamberRecommender(BaseRecommender):
    """Echo chamber recommender (for research purposes).
    
    This algorithm intentionally creates echo chambers by recommending
    content that strongly aligns with the user's existing opinions.
    Useful for studying the effects of echo chambers on social dynamics.
    """
    
    def __init__(
        self,
        max_recommendations: int = 10,
        similarity_bias: float = 0.9,
    ):
        super().__init__(max_recommendations)
        self.similarity_bias = similarity_bias
    
    def recommend(
        self,
        user: User,
        posts: Sequence[Post],
        interactions: Sequence[Interaction],
        all_users: Sequence[User],
    ) -> List[str]:
        available_posts = self._filter_own_posts(user, posts)
        if not available_posts:
            return []
        
        # Score posts by similarity to user's opinions/interests
        post_scores = []
        for post in available_posts:
            if user.embedding is not None and post.embedding is not None:
                similarity = self._compute_cosine_similarity(user.embedding, post.embedding)
            elif user.opinion_vector is not None and post.embedding is not None:
                similarity = self._compute_cosine_similarity(user.opinion_vector, post.embedding)
            else:
                similarity = random.random()
            
            # Apply bias to strongly prefer similar content
            biased_score = similarity ** (1 / self.similarity_bias)
            post_scores.append((post.post_id, biased_score))
        
        post_scores.sort(key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in post_scores[:self.max_recommendations]]


class HybridRecommender(BaseRecommender):
    """Hybrid recommender combining multiple strategies.
    
    This recommender combines multiple recommendation strategies
    with configurable weights to balance different objectives.
    """
    
    def __init__(
        self,
        max_recommendations: int = 10,
        collaborative_weight: float = 0.3,
        bridging_weight: float = 0.3,
        diversity_weight: float = 0.2,
        random_weight: float = 0.2,
    ):
        super().__init__(max_recommendations)
        
        # Normalize weights
        total = collaborative_weight + bridging_weight + diversity_weight + random_weight
        self.collaborative_weight = collaborative_weight / total
        self.bridging_weight = bridging_weight / total
        self.diversity_weight = diversity_weight / total
        self.random_weight = random_weight / total
        
        # Initialize sub-recommenders
        self.collaborative = CollaborativeFilteringRecommender(max_recommendations)
        self.bridging = BridgingRecommender(max_recommendations)
        self.diversity = DiversityRecommender(max_recommendations)
        self.random = RandomRecommender(max_recommendations)
    
    def recommend(
        self,
        user: User,
        posts: Sequence[Post],
        interactions: Sequence[Interaction],
        all_users: Sequence[User],
    ) -> List[str]:
        # Get recommendations from each sub-recommender
        collab_recs = self.collaborative.recommend(user, posts, interactions, all_users)
        bridge_recs = self.bridging.recommend(user, posts, interactions, all_users)
        diverse_recs = self.diversity.recommend(user, posts, interactions, all_users)
        random_recs = self.random.recommend(user, posts, interactions, all_users)
        
        # Compute how many items to take from each
        collab_count = int(self.max_recommendations * self.collaborative_weight)
        bridge_count = int(self.max_recommendations * self.bridging_weight)
        diverse_count = int(self.max_recommendations * self.diversity_weight)
        random_count = self.max_recommendations - collab_count - bridge_count - diverse_count
        
        # Combine recommendations, removing duplicates
        combined: List[str] = []
        seen: set = set()
        
        for rec_list, count in [
            (collab_recs, collab_count),
            (bridge_recs, bridge_count),
            (diverse_recs, diverse_count),
            (random_recs, random_count),
        ]:
            for post_id in rec_list:
                if post_id not in seen and len(combined) < self.max_recommendations:
                    combined.append(post_id)
                    seen.add(post_id)
                if len(combined) >= self.max_recommendations:
                    break
        
        return combined


def create_recommender(
    rec_type: RecommendationType | str,
    max_recommendations: int = 10,
    **kwargs,
) -> BaseRecommender:
    """Factory function to create a recommender instance.
    
    Args:
        rec_type: Type of recommender to create.
        max_recommendations: Maximum posts to recommend per user.
        **kwargs: Additional arguments passed to the recommender constructor.
        
    Returns:
        An instance of the requested recommender.
        
    Example:
        >>> recommender = create_recommender("bridging", max_recommendations=15)
        >>> recommendations = recommender.recommend(user, posts, interactions, all_users)
    """
    if isinstance(rec_type, str):
        rec_type = RecommendationType(rec_type.lower())
    
    recommender_map = {
        RecommendationType.RANDOM: RandomRecommender,
        RecommendationType.COLLABORATIVE: CollaborativeFilteringRecommender,
        RecommendationType.BRIDGING: BridgingRecommender,
        RecommendationType.DIVERSITY: DiversityRecommender,
        RecommendationType.ECHO_CHAMBER: EchoChamberRecommender,
        RecommendationType.HYBRID: HybridRecommender,
    }
    
    recommender_class = recommender_map.get(rec_type, RandomRecommender)
    return recommender_class(max_recommendations=max_recommendations, **kwargs)
