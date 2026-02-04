"""Content moderation and recommendation algorithms for the simulation.

This module provides various recommendation algorithms that can be used
to control how content is distributed to agents in the simulation.
These algorithms aim to study the effects of different content distribution
strategies on social dynamics, including echo chamber formation and polarization.

Available Algorithms:
    - RandomRecommender: Random baseline
    - CollaborativeFilteringRecommender: User-based collaborative filtering
    - BridgingRecommender: Promotes content that bridges opinion clusters
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import random
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class RecommendationType(str, Enum):
    """Types of recommendation algorithms available."""
    
    RANDOM = "random"
    COLLABORATIVE = "collaborative"
    BRIDGING = "bridging"


class Post(BaseModel):
    """Represents a post in the simulation."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
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
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    user_id: int
    bio: str = ""
    embedding: Optional[np.ndarray] = None
    interests: List[str] = Field(default_factory=list)
    opinion_vector: Optional[np.ndarray] = None  # Opinion on various topics
    num_followers: int = 0


class Interaction(BaseModel):
    """Represents a user interaction with a post."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
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

    # Some platforms may call refresh hooks; provide no-op defaults for compatibility.
    async def refresh_cache(self, *args, **kwargs) -> None:  # pragma: no cover - integration hook
        return None

    def refresh(self, *args, **kwargs) -> None:  # pragma: no cover - integration hook
        return None
    
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
    To avoid mode collapse (everyone seeing the same post), we mix in:
    - recency weighting (fresh content is preferred),
    - inverse-frequency dampening (popular posts get penalized),
    - a small exploration budget (random unseen posts).
    """
    
    def __init__(
        self, 
        max_recommendations: int = 10,
        similarity_threshold: float = 0.5,
        k_neighbors: int = 5,
        explore_ratio: float = 0.3,
        recency_half_life: float = 1000.0,
        sampling_temperature: float = 0.7,
        mmr_lambda: float = 0.75,
        mmr_candidates: int = 50,
    ):
        super().__init__(max_recommendations)
        self.similarity_threshold = similarity_threshold
        self.k_neighbors = k_neighbors
        self.explore_ratio = explore_ratio
        self.recency_half_life = recency_half_life
        self.sampling_temperature = sampling_temperature
        self.mmr_lambda = mmr_lambda
        self.mmr_candidates = mmr_candidates
        self._liked_by_user: Dict[int, List[str]] = {}
        self._post_latest_ts: Dict[str, float] = {}
        self._now_ts: float = 0.0
        self._prepared_sig: Tuple[int, int] | None = None

    def _prepare(
        self,
        interactions: Sequence[Interaction],
        all_users: Sequence[User],
    ) -> None:
        last_sig = None
        if interactions:
            last = interactions[-1]
            last_sig = (last.user_id, last.post_id, last.action, last.timestamp)

        sig = (len(interactions), last_sig, len(all_users))
        if self._prepared_sig == sig:
            return

        liked_by_user: Dict[int, List[str]] = {}
        post_latest_ts: Dict[str, float] = {}
        max_ts = 0.0
        for interaction in interactions:
            if interaction.action in ("like", "repost"):
                liked_by_user.setdefault(interaction.user_id, []).append(interaction.post_id)
            try:
                # Store latest timestamp as float (time_step-like in this sim)
                ts_val = float(interaction.timestamp)
                if ts_val > max_ts:
                    max_ts = ts_val
                prev = post_latest_ts.get(interaction.post_id, 0.0)
                if ts_val > prev:
                    post_latest_ts[interaction.post_id] = ts_val
            except Exception:
                continue

        self._liked_by_user = liked_by_user
        self._post_latest_ts = post_latest_ts
        self._now_ts = max_ts
        self._prepared_sig = sig
    
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
        
        self._prepare(interactions, all_users)

        # Find similar users based on embedding or opinion vector
        similar_users = self._find_similar_users(user, all_users)

        # Score posts based on similar users' interactions
        post_scores: Dict[str, float] = {}
        freq: Dict[str, int] = {}
        for liked_list in self._liked_by_user.values():
            for pid in liked_list:
                freq[pid] = freq.get(pid, 0) + 1

        for similar_user, similarity in similar_users:
            for post_id in self._liked_by_user.get(similar_user.user_id, []):
                if post_id not in post_scores:
                    post_scores[post_id] = 0.0
                freshness = self._compute_recency(post_id)
                pop_penalty = 1.0 / (1.0 + np.log1p(freq.get(post_id, 1)))
                post_scores[post_id] += similarity * freshness * pop_penalty
        
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
        if not scored_posts:
            sample_size = min(self.max_recommendations, len(available_posts))
            selected = random.sample(available_posts, sample_size)
            return [p.post_id for p in selected]

        # Collapse countermeasure 1: soften deterministic top-k with temperature sampling.
        sampled = self._temperature_sample(scored_posts, k=min(self.mmr_candidates, len(scored_posts)))
        sampled_scores = [(pid, post_scores[pid]) for pid in sampled]

        # Collapse countermeasure 2: re-rank for diversity (MMR) when embeddings exist.
        post_embeddings = {p.post_id: p.embedding for p in posts if p.embedding is not None}
        reranked = self._mmr_rerank(sampled_scores, post_embeddings)
        top_k = [pid for pid, _ in reranked[: self.max_recommendations]]

        # Exploration: add a small slice of random unseen posts to fight collapse
        explore_k = max(1, int(self.max_recommendations * self.explore_ratio))
        seen = set(top_k)
        unseen_candidates = [p.post_id for p in available_posts if p.post_id not in seen]
        if unseen_candidates and explore_k > 0:
            explore_sample = random.sample(unseen_candidates, min(explore_k, len(unseen_candidates)))
            top_k = (top_k + explore_sample)[: self.max_recommendations]
        return top_k
    
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

    def _temperature_sample(self, scored_posts: Sequence[Tuple[str, float]], k: int) -> List[str]:
        """Sample without replacement using a softmax distribution."""
        if not scored_posts or k <= 0:
            return []
        k = min(k, len(scored_posts))
        scores = np.array([max(0.0, float(s)) for _, s in scored_posts], dtype=float)
        if np.allclose(scores, 0):
            ids = [pid for pid, _ in scored_posts]
            random.shuffle(ids)
            return ids[:k]

        temp = max(1e-6, float(self.sampling_temperature))
        logits = scores / temp
        logits = logits - np.max(logits)
        weights = np.exp(logits)
        probs = weights / weights.sum()

        chosen: list[str] = []
        candidates = list(scored_posts)
        for _ in range(k):
            if not candidates:
                break
            cand_ids = [pid for pid, _ in candidates]
            cand_scores = np.array([max(0.0, float(s)) for _, s in candidates], dtype=float)
            if np.allclose(cand_scores, 0):
                random.shuffle(cand_ids)
                chosen.extend(cand_ids[: (k - len(chosen))])
                break
            logits = cand_scores / temp
            logits = logits - np.max(logits)
            w = np.exp(logits)
            p = w / w.sum()
            idx = int(np.random.choice(len(candidates), p=p))
            chosen.append(candidates[idx][0])
            del candidates[idx]
        return chosen[:k]

    def _mmr_rerank(
        self,
        candidates: Sequence[Tuple[str, float]],
        post_embeddings: Dict[str, np.ndarray],
    ) -> List[Tuple[str, float]]:
        """Maximal Marginal Relevance reranking to reduce near-duplicates."""
        if not candidates:
            return []
        if self.mmr_lambda >= 0.999:
            return list(sorted(candidates, key=lambda x: x[1], reverse=True))

        lambda_val = float(np.clip(self.mmr_lambda, 0.0, 1.0))
        remaining = list(sorted(candidates, key=lambda x: x[1], reverse=True))
        selected: list[Tuple[str, float]] = []

        def _sim(a: str, b: str) -> float:
            va = post_embeddings.get(a)
            vb = post_embeddings.get(b)
            if va is None or vb is None:
                return 0.0
            return float(self._compute_cosine_similarity(va, vb))

        while remaining and len(selected) < min(self.max_recommendations, len(remaining)):
            if not selected:
                selected.append(remaining.pop(0))
                continue

            best_idx = 0
            best_score = -float("inf")
            for idx, (pid, base_score) in enumerate(remaining):
                max_sim = max(_sim(pid, sid) for sid, _ in selected)
                mmr_score = lambda_val * float(base_score) - (1.0 - lambda_val) * max_sim
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            selected.append(remaining.pop(best_idx))

        # Append any leftovers (already sorted) to keep a total ordering.
        selected.extend(remaining)
        return selected

    def _compute_recency(self, post_id: str) -> float:
        """Exponential decay based on simulation time_step, not wall-clock time."""
        ts = self._post_latest_ts.get(post_id)
        if ts is None:
            return 1.0
        try:
            now = float(self._now_ts)
            if now <= 0:
                return 1.0
            # Timestamps are time_step-like small integers in this simulation.
            age = max(0.0, now - float(ts))
            decay = 0.5 ** (age / self.recency_half_life) if self.recency_half_life > 0 else 1.0
            return float(decay)
        except Exception:
            return 1.0


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
    }

    if rec_type not in recommender_map:
        raise ValueError(
            f"Unsupported recommendation type: {rec_type}. "
            f"Supported types are: {[r.value for r in recommender_map]}"
        )

    recommender_class = recommender_map[rec_type]
    return recommender_class(max_recommendations=max_recommendations, **kwargs)
