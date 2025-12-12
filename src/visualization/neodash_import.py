"""Import NeoDash dashboard configuration into Neo4j."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from neo4j import GraphDatabase, basic_auth


DEFAULT_DASHBOARD_PATH = Path("data/neodash/dashboard.json")
DEFAULT_NEO4J_URI = "bolt://localhost:7687"
DEFAULT_NEO4J_USER = "neo4j"
DEFAULT_NEO4J_PASSWORD = "neo4j1234"


def load_dashboard(dashboard_path: Path) -> dict:
    """Load dashboard JSON from file."""
    with dashboard_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_dashboard_to_neo4j(
    dashboard: dict,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    dashboard_name: str = "AI Agent Simulation Dashboard",
) -> None:
    """Save dashboard configuration to Neo4j as a node.
    
    NeoDash can load dashboards from Neo4j using the _Neodash_Dashboard node.
    """
    driver = GraphDatabase.driver(neo4j_uri, auth=basic_auth(neo4j_user, neo4j_password))
    
    # Generate a UUID for the dashboard if not present
    dashboard_uuid = dashboard.get("uuid") or str(uuid.uuid4())
    dashboard["uuid"] = dashboard_uuid
    
    try:
        with driver.session() as session:
            # Delete existing dashboard with same title or uuid
            session.run(
                "MATCH (d:_Neodash_Dashboard) WHERE d.title = $title OR d.uuid = $uuid DETACH DELETE d",
                title=dashboard_name,
                uuid=dashboard_uuid,
            )
            
            # Create new dashboard node with uuid
            dashboard_json = json.dumps(dashboard, ensure_ascii=False)
            session.run(
                """
                CREATE (d:_Neodash_Dashboard {
                    title: $title,
                    uuid: $uuid,
                    date: datetime(),
                    user: $user,
                    content: $content
                })
                """,
                title=dashboard_name,
                uuid=dashboard_uuid,
                user=neo4j_user,
                content=dashboard_json,
            )
            
            print(f"✅ Dashboard '{dashboard_name}' saved to Neo4j")
            print(f"   UUID: {dashboard_uuid}")
            print(f"   Open NeoDash at http://localhost:5005")
            print(f"   Click 'Load' -> 'Load from Neo4j' to access the dashboard")
    finally:
        driver.close()


def list_saved_queries(queries_dir: Path) -> None:
    """List all saved Cypher query files."""
    if not queries_dir.exists():
        print("No query files found.")
        return
    
    print("\n📋 Available Cypher Query Files:")
    print("-" * 50)
    for query_file in sorted(queries_dir.glob("*.cypher")):
        print(f"  - {query_file.name}")
    print("-" * 50)
    print(f"Location: {queries_dir.absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description="Import NeoDash dashboard into Neo4j",
    )
    parser.add_argument(
        "--dashboard-path",
        type=Path,
        default=DEFAULT_DASHBOARD_PATH,
        help=f"Path to dashboard JSON file (default: {DEFAULT_DASHBOARD_PATH})",
    )
    parser.add_argument(
        "--neo4j-uri",
        default=DEFAULT_NEO4J_URI,
        help=f"Neo4j bolt URI (default: {DEFAULT_NEO4J_URI})",
    )
    parser.add_argument(
        "--neo4j-user",
        default=DEFAULT_NEO4J_USER,
        help=f"Neo4j username (default: {DEFAULT_NEO4J_USER})",
    )
    parser.add_argument(
        "--neo4j-password",
        default=DEFAULT_NEO4J_PASSWORD,
        help=f"Neo4j password (default: {DEFAULT_NEO4J_PASSWORD})",
    )
    parser.add_argument(
        "--dashboard-name",
        default="AI Agent Simulation Dashboard",
        help="Dashboard name in Neo4j",
    )
    parser.add_argument(
        "--list-queries",
        action="store_true",
        help="List available Cypher query files",
    )
    
    args = parser.parse_args()
    
    if args.list_queries:
        queries_dir = args.dashboard_path.parent / "queries"
        list_saved_queries(queries_dir)
        return
    
    if not args.dashboard_path.exists():
        print(f"❌ Dashboard file not found: {args.dashboard_path}")
        return
    
    print(f"Loading dashboard from: {args.dashboard_path}")
    dashboard = load_dashboard(args.dashboard_path)
    
    print(f"Connecting to Neo4j at: {args.neo4j_uri}")
    save_dashboard_to_neo4j(
        dashboard=dashboard,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        dashboard_name=args.dashboard_name,
    )


if __name__ == "__main__":
    main()
