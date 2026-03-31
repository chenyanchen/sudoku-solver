"""Simulate active-learning rounds from Label Studio project data."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def simulate_rounds(
    *,
    db_path: Path,
    project_id: int,
    model_version: str | None = None,
    batch_size: int = 5,
) -> dict[str, object]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    project = conn.execute(
        "select model_version from project where id = ?",
        (project_id,),
    ).fetchone()
    if project is None:
        raise RuntimeError(f"Project {project_id} not found in {db_path}")
    effective_version = model_version or project["model_version"]

    annotated_rows = conn.execute(
        """
        select distinct t.id
        from task t
        join task_completion tc on tc.task_id = t.id
        where t.project_id = ?
        order by t.id
        """,
        (project_id,),
    ).fetchall()
    annotated_task_ids = [row["id"] for row in annotated_rows]

    candidate_rows = conn.execute(
        """
        with latest_predictions as (
          select
            p.task_id,
            p.score,
            p.model_version,
            row_number() over (
              partition by p.task_id
              order by p.created_at desc, p.id desc
            ) as rn
          from prediction p
          join task t on t.id = p.task_id
          where t.project_id = ?
            and p.model_version = ?
        )
        select task_id, score
        from latest_predictions
        where rn = 1
          and task_id not in (
            select tc.task_id
            from task_completion tc
            join task t2 on t2.id = tc.task_id
            where t2.project_id = ?
          )
        order by score asc, task_id asc
        """,
        (project_id, effective_version, project_id),
    ).fetchall()
    conn.close()

    candidates = [{"task_id": row["task_id"], "score": row["score"]} for row in candidate_rows]
    rounds: list[dict[str, object]] = []
    current_gold = list(annotated_task_ids)

    for index in range(0, len(candidates), batch_size):
        batch = candidates[index:index + batch_size]
        if not batch:
            break
        rounds.append(
            {
                "round": len(rounds) + 1,
                "selected": batch,
                "gold_train_task_ids_after_round": current_gold + [item["task_id"] for item in batch],
            }
        )
        current_gold = rounds[-1]["gold_train_task_ids_after_round"]

    return {
        "project_id": project_id,
        "model_version": effective_version,
        "initial_gold_task_ids": annotated_task_ids,
        "candidate_count": len(candidates),
        "batch_size": batch_size,
        "rounds": rounds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate active-learning annotation rounds")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path.home() / "data" / "label-studio" / "label_studio.sqlite3",
    )
    parser.add_argument("--project-id", type=int, default=2)
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--batch-size", type=int, default=5)
    args = parser.parse_args()

    payload = simulate_rounds(
        db_path=args.db_path,
        project_id=args.project_id,
        model_version=args.model_version,
        batch_size=args.batch_size,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
