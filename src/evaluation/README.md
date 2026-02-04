# Evaluation: ECS / EchoGAE

このフォルダで使っている ECS と EchoGAE は、**SQLite のどのテーブルを使って計算しているか**が重要なので、現状実装に即して整理します。

## 共通の前提: 「いいね」グラフ
- **ノード**: `user` テーブルの `user_id`
- **エッジ**: `like` と `post` を結合し、
  - `like.user_id`（いいねしたユーザー）
  - `post.user_id`（投稿者）
  の **ユーザー間の無向エッジ**を作ります
- **重み**: 同じ組み合わせのいいね回数の合計
- **自己エッジ除外**: 自分の投稿へのいいねは除外

実装: `src/evaluation/graph_data.py::load_like_graph_from_connection`

使用テーブル
- `user`
- `post`
- `like`

---

## ECS (Echo Chamber Score)
ECS は **ユーザー埋め込み + いいねグラフのコミュニティ**で計算しています。

### 入力データ
1. **コミュニティ検出（グラフ構造）**
   - いいねグラフ（上記）を使用
   - 検出方法: Leiden → Louvain → Greedy の順にフォールバック
   - 実装: `src/evaluation/ecs.py::detect_communities_from_likes`

2. **ユーザー埋め込み**
   - `user_embedding` テーブルの `embedding`
   - 実装: `src/evaluation/ecs.py::load_user_embeddings`

### ECS 計算の流れ
- 同じコミュニティ内の距離（凝集）と、他コミュニティとの距離（分離）から s(u) を計算
- コミュニティごとの平均が ECS*(ω)
- コミュニティ平均の平均が ECS(Ω)

実装: `src/evaluation/ecs.py::compute_ecs`

### 重要な注意点
- `user_embedding` が **ない/空** だと ECS は 0 扱いでスキップされます。
- **投稿本文そのものは使いません**（使うのは `user_embedding` だけ）

---

## EchoGAE
EchoGAE は **いいねグラフ + ユーザー特徴量**でグラフ自己符号化器を学習します。

### 入力データ
1. **グラフ構造（隣接行列）**
   - いいねグラフ（上記）を使用
   - `min_edge_weight` 以上の重みだけ採用
   - 実装: `src/evaluation/echo_pipeline.py::_build_adjacency`

2. **ユーザー特徴量（feature matrix）**
   以下の平均を使います。
   - `user_embedding`（ユーザーの埋め込み）
   - **投稿・コメント埋め込みの平均**（ユーザー単位に集約）

   実装: `src/evaluation/echo_pipeline.py::_build_feature_matrix`

### 投稿/コメント埋め込みの集約方法
- `post_embedding` と `comment_embedding` を `post` / `comment` に結合
- **ユーザーごとに平均**を取って feature に利用

実装: `src/evaluation/echo_pipeline.py::load_user_content_embeddings`

使用テーブル
- `user_embedding`
- `post_embedding` + `post`
- `comment_embedding` + `comment`

### 重要な注意点
- `user_embedding` と `post/comment_embedding` の両方が無い場合、
  **ランダム初期化ベクトル**で特徴量を埋めます
  - これは `feature_dim`（デフォルト128）次元の正規乱数
- **投稿本文の生テキストは直接使いません**
  - 使うのは埋め込み（embedding）だけです

---

## まとめ（現状の実装に基づく）
- **ECS**: `like` グラフ + `user_embedding`
- **EchoGAE**: `like` グラフ + (`user_embedding` と `post/comment_embedding` の平均)
- **本文テキスト**は直接使われず、**embedding テーブル経由**のみ

実装参照
- ECS: `src/evaluation/ecs.py`
- EchoGAE: `src/evaluation/echo_pipeline.py`, `src/evaluation/echogae.py`
- いいねグラフ: `src/evaluation/graph_data.py`
