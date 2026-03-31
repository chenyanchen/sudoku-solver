from __future__ import annotations

import sqlite3
from pathlib import Path

from scripts.simulate_active_learning_rounds import simulate_rounds


def _init_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        create table project (id integer primary key, model_version text);
        create table task (
          id integer primary key,
          project_id integer,
          data text
        );
        create table prediction (
          id integer primary key,
          task_id integer,
          project_id integer,
          model_version text,
          score real,
          result text,
          created_at text
        );
        create table task_completion (
          id integer primary key,
          task_id integer,
          result text,
          updated_at text
        );
        """
    )
    conn.execute("insert into project(id, model_version) values(2, '20260331.4')")
    for task_id in range(1, 8):
        conn.execute(
            "insert into task(id, project_id, data) values(?, 2, '{}')",
            (task_id,),
        )
    conn.execute(
        "insert into task_completion(id, task_id, result, updated_at) values(1, 1, '[]', '2026-03-31 09:00:00')"
    )
    conn.execute(
        "insert into task_completion(id, task_id, result, updated_at) values(2, 2, '[]', '2026-03-31 09:00:00')"
    )
    scores = {
        3: 0.20,
        4: 0.10,
        5: 0.70,
        6: 0.50,
        7: 0.30,
    }
    for idx, (task_id, score) in enumerate(scores.items(), start=10):
        conn.execute(
            """
            insert into prediction(id, task_id, project_id, model_version, score, result, created_at)
            values(?, ?, 2, '20260331.4', ?, '[]', '2026-03-31 10:00:00')
            """,
            (idx, task_id, score),
        )
    conn.commit()
    conn.close()


def test_simulate_rounds_selects_lowest_score_unlabeled_tasks(tmp_path):
    db_path = tmp_path / "label_studio.sqlite3"
    _init_db(db_path)

    payload = simulate_rounds(
        db_path=db_path,
        project_id=2,
        batch_size=2,
    )

    assert payload["initial_gold_task_ids"] == [1, 2]
    assert payload["candidate_count"] == 5
    assert payload["rounds"] == [
        {
            "round": 1,
            "selected": [{"task_id": 4, "score": 0.1}, {"task_id": 3, "score": 0.2}],
            "gold_train_task_ids_after_round": [1, 2, 4, 3],
        },
        {
            "round": 2,
            "selected": [{"task_id": 7, "score": 0.3}, {"task_id": 6, "score": 0.5}],
            "gold_train_task_ids_after_round": [1, 2, 4, 3, 7, 6],
        },
        {
            "round": 3,
            "selected": [{"task_id": 5, "score": 0.7}],
            "gold_train_task_ids_after_round": [1, 2, 4, 3, 7, 6, 5],
        },
    ]
