"""
SQLite 持久化存储层
支持 Experience、Skill、Failure 三张表的 CRUD
"""
import sqlite3
import json
import os
from datetime import datetime
from ..schemas.models import ExperienceRecord, SkillTemplate, FailureNote


class SQLiteStore:
    """
    使用 SQLite 作为经验池、技能库和失败记录的统一存储后端。
    相比 JSONL，SQLite 提供更强的查询能力、事务支持和扩展性。
    """

    def __init__(self, db_path: str = "experiments/learning.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_tables()

    def _init_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_request TEXT NOT NULL,
                    env_state TEXT,
                    agent_actions TEXT,
                    user_feedback TEXT,
                    timestamp TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS skills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    params TEXT,
                    trigger_patterns TEXT,
                    action_template TEXT,
                    success_rate REAL DEFAULT 0.0,
                    usage_count INTEGER DEFAULT 0,
                    version INTEGER DEFAULT 1,
                    created_at TEXT,
                    updated_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS failures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    skill_name TEXT NOT NULL,
                    failure_pattern TEXT,
                    root_cause TEXT,
                    timestamp TEXT
                )
                """
            )
            conn.commit()

    # ----------------- Experience -----------------
    def add_experience(self, record: ExperienceRecord) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO experiences (session_id, user_request, env_state, agent_actions, user_feedback, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.session_id,
                    record.user_request,
                    json.dumps(record.env_state, ensure_ascii=False),
                    json.dumps(record.agent_actions, ensure_ascii=False),
                    record.user_feedback,
                    record.timestamp.isoformat(),
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_experiences(self, limit: int = 1000, offset: int = 0) -> list[ExperienceRecord]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT id, session_id, user_request, env_state, agent_actions, user_feedback, timestamp FROM experiences ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
            rows = cursor.fetchall()
            return [self._row_to_experience(row) for row in rows]

    def _row_to_experience(self, row) -> ExperienceRecord:
        return ExperienceRecord(
            id=row[0],
            session_id=row[1],
            user_request=row[2],
            env_state=json.loads(row[3]) if row[3] else {},
            agent_actions=json.loads(row[4]) if row[4] else [],
            user_feedback=row[5],
            timestamp=datetime.fromisoformat(row[6]),
        )

    # ----------------- Skill -----------------
    def add_skill(self, skill: SkillTemplate) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO skills (name, description, params, trigger_patterns, action_template, success_rate, usage_count, version, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    description=excluded.description,
                    params=excluded.params,
                    trigger_patterns=excluded.trigger_patterns,
                    action_template=excluded.action_template,
                    success_rate=excluded.success_rate,
                    usage_count=excluded.usage_count,
                    version=excluded.version,
                    updated_at=excluded.updated_at
                """,
                (
                    skill.name,
                    skill.description,
                    json.dumps(skill.params, ensure_ascii=False),
                    json.dumps(skill.trigger_patterns, ensure_ascii=False),
                    json.dumps(skill.action_template, ensure_ascii=False),
                    skill.success_rate,
                    skill.usage_count,
                    skill.version,
                    skill.created_at.isoformat(),
                    skill.updated_at.isoformat(),
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_skill(self, name: str) -> SkillTemplate | None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM skills WHERE name = ?", (name,))
            row = cursor.fetchone()
            return self._row_to_skill(row) if row else None

    def list_skills(self) -> list[SkillTemplate]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM skills ORDER BY updated_at DESC")
            return [self._row_to_skill(row) for row in cursor.fetchall()]

    def _row_to_skill(self, row) -> SkillTemplate:
        return SkillTemplate(
            id=row[0],
            name=row[1],
            description=row[2],
            params=json.loads(row[3]) if row[3] else [],
            trigger_patterns=json.loads(row[4]) if row[4] else [],
            action_template=json.loads(row[5]) if row[5] else {},
            success_rate=row[6],
            usage_count=row[7],
            version=row[8],
            created_at=datetime.fromisoformat(row[9]),
            updated_at=datetime.fromisoformat(row[10]),
        )

    # ----------------- Failure -----------------
    def add_failure(self, note: FailureNote) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO failures (skill_name, failure_pattern, root_cause, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (note.skill_name, note.failure_pattern, note.root_cause, note.timestamp.isoformat()),
            )
            conn.commit()
            return cursor.lastrowid

    def get_failures(self, skill_name: str) -> list[FailureNote]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM failures WHERE skill_name = ? ORDER BY timestamp DESC",
                (skill_name,),
            )
            rows = cursor.fetchall()
            return [self._row_to_failure(row) for row in rows]

    def _row_to_failure(self, row) -> FailureNote:
        return FailureNote(
            id=row[0],
            skill_name=row[1],
            failure_pattern=row[2],
            root_cause=row[3],
            timestamp=datetime.fromisoformat(row[4]),
        )
