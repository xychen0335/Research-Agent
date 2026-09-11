"""Create a tiny deterministic fixture; these tasks are not a benchmark."""

import json
import sqlite3
from pathlib import Path


def create_demo(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "shop.sqlite"
    if not path.exists():
        conn = sqlite3.connect(path)
        try:
            conn.executescript("""
                CREATE TABLE users (id INTEGER PRIMARY KEY, city TEXT);
                CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER,
                    amount REAL, status TEXT, created_at TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(id));
                INSERT INTO users VALUES (1,'深圳'),(2,'深圳'),(3,'北京'),(4,'北京'),(5,'上海');
                INSERT INTO orders VALUES
                    (1,1,100,'paid','2025-01-02'),(2,1,150,'paid','2025-02-03'),
                    (3,2,80,'cancelled','2025-03-01'),(4,3,200,'paid','2025-04-01'),
                    (5,3,50,'paid','2025-04-02'),(6,3,90,'paid','2024-12-31'),
                    (7,4,40,'paid','2025-06-01'),(8,5,300,'paid','2026-01-01');
            """)
            conn.commit()
        finally:
            conn.close()
    tasks = [
        {"id": "paid_total", "question": "统计 2025 年已支付（status='paid'）订单总金额。",
         "gold_sql": "SELECT SUM(amount) FROM orders WHERE status='paid' AND created_at >= '2025-01-01' AND created_at < '2026-01-01'"},
        {"id": "city_totals", "question": "按城市统计 2025 年已支付订单金额，只返回存在这类订单的城市，按总金额降序。",
         "gold_sql": "SELECT u.city,SUM(o.amount) FROM users u JOIN orders o ON u.id=o.user_id WHERE o.status='paid' AND o.created_at >= '2025-01-01' AND o.created_at < '2026-01-01' GROUP BY u.city ORDER BY SUM(o.amount) DESC", "ordered": True},
        {"id": "repeat_users", "question": "返回 2025 年至少有两笔已支付订单的用户 ID。",
         "gold_sql": "SELECT user_id FROM orders WHERE status='paid' AND created_at >= '2025-01-01' AND created_at < '2026-01-01' GROUP BY user_id HAVING COUNT(*) >= 2"},
        {"id": "no_orders", "question": "返回 2025 年没有已支付订单的用户 ID，包括有订单但未支付的用户。",
         "gold_sql": "SELECT id FROM users WHERE id NOT IN (SELECT user_id FROM orders WHERE status='paid' AND created_at >= '2025-01-01' AND created_at < '2026-01-01')"},
    ]
    for task in tasks:
        task["database"] = str(path.resolve())
    task_path = directory / "tasks.jsonl"
    task_path.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks), encoding="utf-8")
    return task_path


if __name__ == "__main__":
    print(create_demo("data/generated"))
