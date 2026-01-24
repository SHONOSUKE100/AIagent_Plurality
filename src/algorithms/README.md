## 推薦アルゴリズム概要

本ディレクトリでは、シミュレーション内で利用するコンテンツ推薦アルゴリズムを実装しています。現在サポートしているのは以下の3種類です。

| タイプ | 目的 | 主な入力 | 期待される効果 |
|--------|------|---------|----------------|
| `random` | ベースライン。無作為に投稿を提示 | ユーザー自身の投稿以外の全投稿 | 最低限の比較対象として利用 |
| `collaborative` | 協調フィルタリング（類似ユーザーの好みを模倣） | ユーザー/投稿埋め込み or 意見ベクトル、過去の「いいね」「リポスト」履歴 | 類似ユーザーが好んだ投稿を提示しやすい |
| `bridging` | 分断修復（異なる意見クラスターをつなぐ） | 意見ベクトル、投稿を支持したユーザー集合、埋め込み | 異なる立場から支持を得た投稿を優先し、分断を緩和 |

### 共通の入出力
- 入力: `User`・`Post`・`Interaction` のリスト。`recommend` は「対象ユーザー」「全投稿」「全インタラクション」「全ユーザー」を受け取る。
- 出力: 推薦する `post_id` のリスト（長さは `max_recommendations` 上限）。
- 共通の前処理: 自分が投稿したものは除外 (`BaseRecommender._filter_own_posts`)。

### 各アルゴリズムの詳細

#### 1. RandomRecommender (`random`)
- 手順: 利用可能な投稿からランダムサンプリング。
- パラメータ: `max_recommendations`（上限数）のみ。
- 役割: 影響を測るためのベースライン。埋め込みや履歴に依存しない。
- 疑似処理:
  1. 自分の投稿を除外。
  2. `min(max_recommendations, 残り投稿数)` をランダム抽出。
  3. `post_id` を返す。

#### 2. CollaborativeFilteringRecommender (`collaborative`)
- 目的: 類似ユーザーが好んだ投稿を提示する。
- 類似度計算:
  - ユーザー埋め込みが両方あればコサイン類似度。
  - なければ意見ベクトル同士でコサイン類似度。
  - 類似度が `similarity_threshold` 以上のユーザーのみ採用。
- 推薦スコア:
  - 類似ユーザーが「like」または「repost」した投稿に類似度を加点。
  - スコア順に並べ、上位を返す。データ不足時はランダムにフォールバック。
- 疑似処理:
  1. 自分以外のユーザーとの類似度を計算し、`similarity_threshold` 以上を上位 `k_neighbors` まで採用。
  2. インタラクションから「like/repost」された投稿ごとに、類似ユーザーの類似度を加算（`post_score[post_id] += similarity`）。
  3. 自分の投稿でないものだけ残し、スコア順にソートして上位を返す。
  4. スコアが空ならランダムに抽出。
- 主なパラメータ:
  - `similarity_threshold`（類似ユーザーとみなす最小値、デフォルト0.5）
  - `k_neighbors`（最大近傍数、デフォルト5）
  - `max_recommendations`
- 注意: 類似嗜好を強化するため、エコーチェンバーを促進する可能性がある。

#### 3. BridgingRecommender (`bridging`)
- 目的: 異なる意見クラスターをまたいで支持される投稿を優先し、分断を緩和する。
- スコア設計:
  - `bridging_score`: 投稿を支持したユーザー群の意見ベクトル分散を計算し、多様性が高いほど加点。意見データがなければ小さな乱数。
  - `relevance_score`: ユーザーと投稿の埋め込みコサイン類似度を0〜1に正規化。埋め込みが無ければ0.5。
  - `combined_score = bridging_weight * bridging_score + relevance_weight * relevance_score`
- 疑似処理:
  1. その投稿に対して「like/repost/comment」したユーザー集合を取得（1人のみなら0点）。
  2. 集合内の意見ベクトル分散を計算し、`variance * 2` を1.0でクリップして `bridging_score` とする（多様なほど高得点）。意見が無ければ `random() * 0.5`。
  3. ユーザーと投稿の埋め込み類似度を `(cosine + 1) / 2` で0〜1にし、`relevance_score` とする（埋め込み欠損時は0.5）。
  4. `combined_score` を計算し、スコア順に上位を返す。
- 主なパラメータ:
  - `bridging_weight`（分断修復の重み、デフォルト0.7）
  - `relevance_weight`（嗜好一致度の重み、デフォルト0.3）
  - `max_recommendations`
- 期待効果: 異なる立場から支持された投稿を優先しつつ、最低限の関心一致も考慮。

### ファクトリ関数
- `create_recommender(rec_type, max_recommendations=10, **kwargs)` でインスタンス生成。
- `rec_type` は `random` / `collaborative` / `bridging` のみサポート。その他を指定すると `ValueError` を返す。

### 実装ファイル
- 本文: `contents_moderation.py`
- APIエクスポート: `__init__.py`

### 今後の拡張アイデア
- 埋め込みが無い環境での軽量類似度（キーワードベースなど）の追加。
- 時系列要素（新規性）やポスト品質スコアを組み込んだハイブリッド拡張。
