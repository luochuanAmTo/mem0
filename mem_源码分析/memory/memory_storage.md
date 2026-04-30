`storage.py` 文件实现了一个**基于 SQLite 的本地存储管理器 `SQLiteManager`**，是 Mem0 系统中用于持久化**记忆变更历史**和**短期会话消息**的关键组件

这个类的设计围绕三个核心任务展开：**管理历史记录表**、**管理会话消息表**，以及保证操作**线程安全**。

| 核心组成部分                                                 | 机制与作用                                                   |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| **初始化与表管理** (`__init__`, `_create_*_table`, `_migrate_*_table`) | 在初始化时**自动创建或迁移** `history` 和 `messages` 两张核心数据表，确保数据库模式始终是最新的。 |
| **历史记录操作** (`add_history`, `batch_add_history`, `get_history`) | 提供对记忆**增/改/删事件的完整记录**，支持单条写入和批量写入。 |
| **会话消息管理** (`save_messages`, `get_last_messages`)      | 负责存储对话消息，并采用**滚动窗口机制**，自动为每个会话保留最新的 N 条记录。 |
| **生命周期与线程安全**                                       | 使用 `threading.Lock()` 保护所有数据库写操作，并提供 `reset()` 和 `close()` 方法管理资源。 |





功能测试：

```py
import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# 配置日志输出，方便观察流程
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SQLiteManager:
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._migrate_history_table()
        self._create_history_table()
        self._create_messages_table()

    def _migrate_history_table(self) -> None:
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                cur = self.connection.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='history'")
                if cur.fetchone() is None:
                    self.connection.execute("COMMIT")
                    return
                cur.execute("PRAGMA table_info(history)")
                old_cols = {row[1] for row in cur.fetchall()}
                expected_cols = {
                    "id", "memory_id", "old_memory", "new_memory", "event",
                    "created_at", "updated_at", "is_deleted", "actor_id", "role",
                }
                if old_cols == expected_cols:
                    self.connection.execute("COMMIT")
                    return

                logger.info("检测到旧版 history 表结构，开始迁移...")
                cur.execute("DROP TABLE IF EXISTS history_old")
                cur.execute("ALTER TABLE history RENAME TO history_old")
                cur.execute(
                    """CREATE TABLE history (
                        id           TEXT PRIMARY KEY,
                        memory_id    TEXT,
                        old_memory   TEXT,
                        new_memory   TEXT,
                        event        TEXT,
                        created_at   DATETIME,
                        updated_at   DATETIME,
                        is_deleted   INTEGER,
                        actor_id     TEXT,
                        role         TEXT
                    )"""
                )
                intersecting = list(expected_cols & old_cols)
                if intersecting:
                    cols_csv = ", ".join(intersecting)
                    cur.execute(f"INSERT INTO history ({cols_csv}) SELECT {cols_csv} FROM history_old")
                cur.execute("DROP TABLE history_old")
                self.connection.execute("COMMIT")
                logger.info("history 表迁移成功。")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"迁移失败: {e}")
                raise

    def _create_history_table(self) -> None:
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute(
                    """CREATE TABLE IF NOT EXISTS history (
                        id           TEXT PRIMARY KEY,
                        memory_id    TEXT,
                        old_memory   TEXT,
                        new_memory   TEXT,
                        event        TEXT,
                        created_at   DATETIME,
                        updated_at   DATETIME,
                        is_deleted   INTEGER,
                        actor_id     TEXT,
                        role         TEXT
                    )"""
                )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"创建 history 表失败: {e}")
                raise

    def _create_messages_table(self) -> None:
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute(
                    """CREATE TABLE IF NOT EXISTS messages (
                        id TEXT PRIMARY KEY,
                        session_scope TEXT,
                        role TEXT,
                        content TEXT,
                        name TEXT,
                        created_at DATETIME
                    )"""
                )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"创建 messages 表失败: {e}")
                raise

    def add_history(
        self,
        memory_id: str,
        old_memory: Optional[str],
        new_memory: Optional[str],
        event: str,
        *,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        is_deleted: int = 0,
        actor_id: Optional[str] = None,
        role: Optional[str] = None,
    ) -> None:
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute(
                    """INSERT INTO history (id, memory_id, old_memory, new_memory, event,
                       created_at, updated_at, is_deleted, actor_id, role)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (str(uuid.uuid4()), memory_id, old_memory, new_memory, event,
                     created_at, updated_at, is_deleted, actor_id, role),
                )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"添加历史记录失败: {e}")
                raise

    def batch_add_history(self, records: List[Dict[str, Any]]) -> None:
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.executemany(
                    """INSERT INTO history (id, memory_id, old_memory, new_memory, event,
                       created_at, updated_at, is_deleted, actor_id, role)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [(str(uuid.uuid4()), r["memory_id"], r["old_memory"], r["new_memory"],
                      r["event"], r["created_at"], r["updated_at"], r.get("is_deleted", 0),
                      r["actor_id"], r["role"]) for r in records]
                )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"批量添加历史失败: {e}")
                raise

    def get_history(self, memory_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self.connection.execute(
                """SELECT id, memory_id, old_memory, new_memory, event,
                   created_at, updated_at, is_deleted, actor_id, role
                   FROM history WHERE memory_id = ?
                   ORDER BY created_at ASC, DATETIME(updated_at) ASC""",
                (memory_id,),
            )
            rows = cur.fetchall()
        return [
            {
                "id": r[0], "memory_id": r[1], "old_memory": r[2],
                "new_memory": r[3], "event": r[4], "created_at": r[5],
                "updated_at": r[6], "is_deleted": bool(r[7]),
                "actor_id": r[8], "role": r[9],
            }
            for r in rows
        ]

    def save_messages(self, messages: List[Dict[str, Any]], session_scope: str) -> None:
        if not messages:
            return
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                now = datetime.now(timezone.utc).isoformat()
                for message in messages:
                    self.connection.execute(
                        """INSERT INTO messages (id, session_scope, role, content, name, created_at)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (
                            str(uuid.uuid4()),
                            session_scope,
                            message.get("role"),
                            message.get("content"),
                            message.get("name"),
                            now,
                        ),
                    )
                # 修正：使用 rowid 排序，确保按插入顺序保留最新的 10 条
                self.connection.execute(
                    """DELETE FROM messages WHERE session_scope = ? AND id NOT IN (
                        SELECT id FROM (
                            SELECT id FROM messages WHERE session_scope = ?
                            ORDER BY rowid DESC LIMIT 10
                        )
                    )""",
                    (session_scope, session_scope),
                )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"Failed to save messages: {e}")
                raise

    def get_last_messages(self, session_scope: str, limit: int = 10) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self.connection.execute(
                """SELECT role, content, name, created_at FROM (
                    SELECT role, content, name, created_at
                    FROM messages WHERE session_scope = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                ) ORDER BY created_at ASC""",
                (session_scope, limit),
            )
            rows = cur.fetchall()
        return [
            {"role": r[0], "content": r[1], "name": r[2], "created_at": r[3]}
            for r in rows
        ]

    def reset(self) -> None:
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute("DROP TABLE IF EXISTS history")
                self.connection.execute("DROP TABLE IF EXISTS messages")
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"重置失败: {e}")
                raise
        self._create_history_table()
        self._create_messages_table()

    def close(self) -> None:
        if self.connection:
            self.connection.close()
            self.connection = None

    def __del__(self):
        self.close()


def print_section(title):
    print(f"\n{'='*20} {title} {'='*20}")


def main():
    # 1. 初始化内存数据库
    print_section("初始化 SQLiteManager (内存数据库)")
    manager = SQLiteManager()  # 默认 :memory:，不会产生物理文件
    print("数据库已创建，创建了 `history` 和 `messages`两张表。")

    # 2. 演示单条历史记录添加
    print_section("添加单条历史记录")
    mem_id = "mem-001"
    manager.add_history(
        memory_id=mem_id,
        old_memory=None,
        new_memory="用户喜欢深色模式",
        event="ADD",
        created_at="2025-01-01T10:00:00Z",
        actor_id="user_123",
        role="user"
    )
    print(f"已为记忆 {mem_id} 添加一条创建事件。")

    manager.add_history(
        memory_id=mem_id,
        old_memory="用户喜欢深色模式",
        new_memory="用户喜欢深色模式，并且偏好 Vim 键位",
        event="UPDATE",
        created_at="2025-01-02T15:30:00Z",
        actor_id="assistant",
        role="ai"
    )
    print(f"已为记忆 {mem_id} 添加一条更新事件。")

    # 3. 查询历史记录
    print_section(f"查询记忆 {mem_id} 的完整历史")
    history = manager.get_history(mem_id)
    for entry in history:
        print(f"  事件: {entry['event']}")
        print(f"    旧值: {entry['old_memory']}")
        print(f"    新值: {entry['new_memory']}")
        print(f"    操作者: {entry['actor_id']} ({entry['role']})")
        print(f"    时间: {entry['created_at']}")
        print("    ----")

    # 4. 批量添加历史记录
    print_section("批量添加历史记录")
    records = [
        {
            "memory_id": "mem-002",
            "old_memory": None,
            "new_memory": "用户地址：北京",
            "event": "ADD",
            "created_at": "2025-02-01T09:00:00Z",
            "updated_at": None,
            "actor_id": "user_456",
            "role": "user"
        },
        {
            "memory_id": "mem-002",
            "old_memory": "用户地址：北京",
            "new_memory": "用户地址：上海",
            "event": "UPDATE",
            "created_at": "2025-03-01T11:00:00Z",
            "updated_at": "2025-03-01T11:00:00Z",
            "actor_id": "user_456",
            "role": "user"
        }
    ]
    manager.batch_add_history(records)
    print(f"✓ 批量为 mem-002 添加了 {len(records)} 条历史记录。")

    # 查看 mem-002 的历史
    history2 = manager.get_history("mem-002")
    for entry in history2:
        print(f"  事件: {entry['event']} -> 新值: {entry['new_memory']}")

    # 5. 消息存储与滚动窗口演示
    print_section("消息存储与滑动窗口（保留最新 10 条）")
    session = "chat_session_123"
    # 模拟连续发送 12 条消息
    for i in range(1, 13):
        messages = [{"role": "user" if i % 2 != 0 else "assistant",
                     "content": f"这是第{i}条消息的内容。",
                     "name": "Alice" if i % 2 != 0 else "Bob"}]
        manager.save_messages(messages, session)
    print(f"✓ 已向会话 {session} 发送 12 条消息（触发自动淘汰旧消息）。")

    # 获取最后 10 条消息
    last_msgs = manager.get_last_messages(session, limit=10)
    print(f"当前会话保留的最新 10 条消息：")
    for idx, msg in enumerate(last_msgs, 1):
        print(f"  {idx}. [{msg['role']}] {msg['content']} (发送者: {msg['name']})")

    # 验证早期消息已被删除
    # 通过获取所有消息的方式，但只有10条，且第一条应该是第3条消息的内容
    assert len(last_msgs) == 10
    assert "第1条消息" not in last_msgs[0]["content"]
    print("✓ 确认最早的两条消息已被自动清除，滑动窗口正常工作。")

    # 6. 重置功能演示
    print_section("重置数据库表")
    manager.reset()
    print("✓ 表已重置，历史记录和消息均被清空。")
    history_after = manager.get_history(mem_id)
    msgs_after = manager.get_last_messages(session)
    print(f"  历史记录数: {len(history_after)}，消息数: {len(msgs_after)}")
    print("✓ 所有数据已清除，表结构仍存在。")

    # 7. 关闭连接
    manager.close()
    print_section("完成")
    print("SQLiteManager 演示结束。")


if __name__ == "__main__":
    main()
```

output：

```shell


==================== 初始化 SQLiteManager (内存数据库) ====================
数据库已创建，创建了 `history` 和 `messages`两张表。

==================== 添加单条历史记录 ====================
已为记忆 mem-001 添加一条创建事件。
已为记忆 mem-001 添加一条更新事件。

==================== 查询记忆 mem-001 的完整历史 ====================
  事件: ADD
    旧值: None
    新值: 用户喜欢深色模式
    操作者: user_123 (user)
    时间: 2025-01-01T10:00:00Z
    ----
  事件: UPDATE
    旧值: 用户喜欢深色模式
    新值: 用户喜欢深色模式，并且偏好 Vim 键位
    操作者: assistant (ai)
    时间: 2025-01-02T15:30:00Z
    ----

==================== 批量添加历史记录 ====================
 批量为 mem-002 添加了 2 条历史记录。
  事件: ADD -> 新值: 用户地址：北京
  事件: UPDATE -> 新值: 用户地址：上海

==================== 消息存储与滑动窗口（保留最新 10 条） ====================
已向会话 chat_session_123 发送 12 条消息（触发自动淘汰旧消息）。
当前会话保留的最新 10 条消息：
  1. [user] 这是第3条消息的内容。 (发送者: Alice)
  2. [assistant] 这是第4条消息的内容。 (发送者: Bob)
  3. [user] 这是第5条消息的内容。 (发送者: Alice)
  4. [assistant] 这是第6条消息的内容。 (发送者: Bob)
  5. [user] 这是第7条消息的内容。 (发送者: Alice)
  6. [assistant] 这是第8条消息的内容。 (发送者: Bob)
  7. [user] 这是第9条消息的内容。 (发送者: Alice)
  8. [assistant] 这是第10条消息的内容。 (发送者: Bob)
  9. [user] 这是第11条消息的内容。 (发送者: Alice)
  10. [assistant] 这是第12条消息的内容。 (发送者: Bob)
  
  确认最早的两条消息已被自动清除，滑动窗口正常工作。

==================== 重置数据库表 ====================
表已重置，历史记录和消息均被清空。
  历史记录数: 0，消息数: 0
所有数据已清除，表结构仍存在。


进程已结束，退出代码为 0

```

