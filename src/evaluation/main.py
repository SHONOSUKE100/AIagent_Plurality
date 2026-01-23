from __future__ import annotations

import sys
import os
import sqlite3
from pathlib import Path

# 1. パスの設定
current_dir = Path(os.getcwd())
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# 2. 外部ライブラリの読み込み
try:
    import plotly.graph_objects as go
    from jinja2 import Template
    from src.evaluation.graph_base import compute_basic_metrics
    from src.evaluation.graph_data import load_like_graph_from_connection
except ImportError as e:
    print(f"エラー: ライブラリが足りません。 {e}")

# --- HTMLレポートを生成する関数 (空データ対応) ---
def generate_html_report(metrics, output_path, is_empty=False):
    status_message = "※シミュレーション進行中のため、データがまだ蓄積されていません。" if is_empty else "解析は正常に完了しました。"
    
    html_template = """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8"><title>Echo Chamber Analysis</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="p-10 bg-slate-50">
        <div class="max-w-4xl mx-auto bg-white p-8 shadow-lg border-t-8 border-red-700 rounded">
            <h1 class="text-3xl font-bold mb-2">Echo Chamber Analysis Report</h1>
            <p class="text-sm text-orange-600 mb-8 font-bold">{{ status_message }}</p>

            <div class="grid grid-cols-2 gap-6 mb-10">
                <div class="p-6 bg-red-50 rounded-xl border border-red-100">
                    <p class="text-xs text-red-600 font-bold uppercase tracking-widest">Homophily Index</p>
                    <p class="text-5xl font-black text-gray-900">{{ "%.4f"|format(metrics.homophily) }}</p>
                </div>
                <div class="p-6 bg-blue-50 rounded-xl border border-blue-100">
                    <p class="text-xs text-blue-600 font-bold uppercase tracking-widest">Modularity</p>
                    <p class="text-5xl font-black text-gray-900">{{ "%.4f"|format(metrics.modularity) }}</p>
                </div>
            </div>

            <div class="p-4 bg-gray-100 rounded text-center text-gray-500 text-sm">
                データが蓄積され次第、グラフと数値が更新されます。
            </div>
        </div>
    </body>
    </html>
    """
    template = Template(html_template)
    rendered = template.render(metrics=metrics, status_message=status_message)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(rendered)

# --- 解析・計算関数 (空データ対応) ---
def compute_metrics(sqlite_path: Path) -> None:
    conn = sqlite3.connect(sqlite_path)
    report_path = current_dir / "results" / "echo_chamber_report.html"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # 1. カラム名の自動判定
        cursor = conn.execute("PRAGMA table_info(user)")
        columns = [row[1] for row in cursor.fetchall()]
        target_col = next((c for c in ["category", "persona", "role", "group_id"] if c in columns), "category")

        # 2. データのロード (None対策)
        result = load_like_graph_from_connection(conn, attribute_column=target_col)
        
        # 3. データが空、またはNoneの場合の処理
        if result is None or len(result[0]) == 0:
            print("INFO: データが空です。空のレポートを出力します。")
            empty_metrics = {
                'homophily': 0.0, 'modularity': 0.0, 'density': 0.0,
                'avg_shortest_path_length': 0.0, 'average_clustering': 0.0
            }
            generate_html_report(empty_metrics, report_path, is_empty=True)
        else:
            nodes, edges, node_attrs = result
            metrics = compute_basic_metrics(nodes, edges, node_attributes=node_attrs)
            generate_html_report(metrics, report_path, is_empty=False)
        
        print(f"SUCCESS: Report saved to {report_path}")

    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        conn.close()

def main():
    db_path = current_dir / "results" / "simulation.db"
    if not db_path.exists():
        db_path = current_dir / "simulation.db"
    
    if db_path.exists():
        compute_metrics(db_path)
    else:
        # DBファイルすらない場合も、空のレポートを出す
        print("INFO: DBファイルが見つかりません。待機用レポートを出力します。")
        empty_metrics = {'homophily': 0.0, 'modularity': 0.0}
        report_path = current_dir / "results" / "echo_chamber_report.html"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        # 簡易的に辞書オブジェクトとして渡す
        class SimpleMetrics:
            def __init__(self, d): self.__dict__ = d
            def __getitem__(self, key): return self.__dict__[key]
        generate_html_report(SimpleMetrics(empty_metrics), report_path, is_empty=True)

if __name__ == "__main__":
    main()