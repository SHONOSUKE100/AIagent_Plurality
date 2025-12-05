# AI Agent Plurality Simulation

## 研究目的
近年、SNS上でのエコーチェンバー現象による社会的分断の深刻化が指摘されている。この現象は、異質な意見を持つ市民間の議論を減少させる政治的影響に加え、心理的な分断をもたらす懸念がある。

エコーチェンバー現象の主要因の一つとして、パーソナライズ機能が挙げられる。同機能はユーザーの嗜好に合致したコンテンツを優先的に提示することで顧客満足度を高め、滞在時間を延長させる効果を持つ一方で、前述のような負の側面も孕んでいる。

本研究では、ユーザー体験の向上と社会的分断の抑制という二律背反的な課題に対し、適切なバランスを保つ推薦システムやプロダクトデザインのあり方を探究する。具体的には、AIエージェントを用いたシミュレーションを通じて検証を行う。

AIエージェントを用いる理由は以下の二点である。
第一に、人間を対象とした実験におけるコストや倫理的制約を回避するためである。
第二に、将来的に人々がAIエージェントを介して情報を取得することが一般化すると予想される中で、パーソナライズされたAIエージェント自身がエコーチェンバーの影響を受ける可能性を検証することは、今後のエージェント設計や倫理的ガイドラインの策定において重要であると考えるためである。

## 研究内容

### シミュレーション概要
100体の異なるペルソナ（性格、政治的志向、興味関心）が付与されたAI Agentを作成し、閉じたSNS環境内での相互作用をシミュレーションする。

### 実験プロセス
1. **初期化 (Initialization)**: 
   - 100体のエージェントを生成。各エージェントには異なるバックグラウンドと初期意見ベクトルを持たせる。
2. **初期投稿 (Cold Start)**: 
   - 他者の影響を受けていない状態で、各エージェントが数件の投稿を行う。
3. **インタラクションループ (Interaction Loop)**: 
   - **推薦 (Recommendation)**: 実験対象のアルゴリズムに基づき、各エージェントにタイムライン（投稿リスト）を提示。
   - **反応 (Reaction)**: エージェントは提示された投稿に対し、閲覧・Like・コメント・無視などの行動を選択する。
   - **意見変容 (Update)**: 接触した情報に基づき、エージェントの内部状態（意見）を更新する。
   - **新規投稿 (Post)**: 更新された意見に基づき、新たな投稿を行う。
4. **評価 (Evaluation)**: 
   - 一定ラウンド終了後、ログデータを分析し指標を算出する。

### シミュレーション環境
- **Framework**: [Oasis (Camel-AI)](https://github.com/camel-ai/oasis) - AIエージェントベースの社会シミュレーションライブラリ
- **LLM**: OpenAI GPT-4o / GPT-4o-mini (予定)
- **Language**: Python

## 使い方
- Python 3.12+ と依存ライブラリをインストールする。
   ```bash
   pip install -e .
   ```
- OpenAI の API キーを環境変数 `OPENAI_API_KEY` に設定する。
- Neo4j を Docker Compose で起動する。
   ```bash
   docker compose up neo4j -d
   ```
   デフォルト認証情報は `neo4j/neo4j1234`（`docker-compose.yml` 参照）。停止する場合は `docker compose down`。
- シミュレーションを実行する。
   ```bash
   uv run main.py --persona-path data/persona/persona.json --seed-post-count 20 --llm-rounds 1
   ```
   `results/<timestamp>/` 以下に `simulation.db`、`metadata.yaml`（実行設定）、使用ペルソナのコピーが保存され、`results/index.csv` と `results/latest_run.txt` へも記録される。
- Neo4j に可視化用のノード・エッジを投入する。
   ```bash
   uv run src/visualization/neo4j_export.py --neo4j-password neo4j1234
   ```
   `results/latest_run.txt` で記録された最新のディレクトリから `simulation.db` を参照し、Neo4j へ投入した後は同ディレクトリに `graph.graphml` と `neo4j_export.yaml` が生成される。Neo4j Desktop や Browser で `MATCH (n) RETURN n LIMIT 200;` などを実行するとネットワークが確認できる。
- よく使うコマンドは [mise](https://mise.jdx.dev/) のタスクからも実行できる。
   ```bash
   mise run simulation
   mise run neo4j-up
   mise run neo4j-export
   mise run evaluate
   ```
   引数を変えたい場合は `mise run simulation -- --seed-post-count 50` のように `--` 以降へ渡す。

## データ保管について
- `results/` : 各実験ごとの成果物（`simulation.db`、`metadata.yaml`、GraphML など）や `index.csv`（実行ログ）、`latest_run.txt`（直近実験パス）を格納。
- `data/neo4j/` : Docker Compose で起動した Neo4j のデータ・ログ・プラグインディレクトリ。自動生成されるため手動編集は不要。

### 検討する推薦アルゴリズム
本研究では、以下のアルゴリズムによる社会動態の違いを比較検証する。

1. **ランダム (Random)**: 
   - ベースライン。無作為にコンテンツを提示。
2. **協調フィルタリング (Collaborative Filtering)**: 
   - 類似した嗜好を持つユーザーの行動履歴に基づく推薦。エコーチェンバーを促進する可能性がある。
3. **Bridging-Based (分断修復型)**: 
   - 異なる意見クラスター間をつなぐ（両者から支持される）コンテンツを優先的に提示。
4. **LLMによる情報提供 (LLM-driven)**: 
   - エージェントの視野を広げるよう、LLMが動的にコンテンツを選定・生成して提示。

## 評価指標

### 1. 情報接触の偏り (Exposure)
- **Intra-List Similarity (ILS)**: 推薦リスト内のコンテンツ類似度。高いほど情報が均質化していることを示す。
- **Category Entropy**: 接触した情報のカテゴリ分布のエントロピー。

### 2. 意見の極性化 (Polarization)
- **Opinion Drift**: エージェントの意見ベクトルが初期値からどれだけ移動したか。
- **Polarization Index**: 集団全体の意見分布の分散や双峰性（Bimodality）を測定。

### 3. 構造的分断 (Segregation)
- **Modularity**: エージェント間の「いいね」やフォロー関係からグラフを構築し、コミュニティの分断度合いを測る。
- **Echo Chamber Score**: 自分と似た意見のコンテンツに接触した割合 vs 異なる意見に接触した割合。

## フォルダ構成 (予定)
```text
.
├── data
├── main.py
├── pyproject.toml
├── README.md
└── src
    ├── agents
    ├── algorithms
    ├── simulation
    └── utils
```