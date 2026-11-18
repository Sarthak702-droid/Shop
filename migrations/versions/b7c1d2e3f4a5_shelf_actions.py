"""shelf actions: per-event action segments and reviewer action feedback

Revision ID: b7c1d2e3f4a5
Revises: e8d95c8244d9
Create Date: 2026-11-18 18:30:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "b7c1d2e3f4a5"
down_revision: Union[str, Sequence[str], None] = "e8d95c8244d9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("events", schema=None) as batch_op:
        batch_op.add_column(sa.Column("actions", sa.JSON(), nullable=False, server_default="[]"))
    with op.batch_alter_table("reviews", schema=None) as batch_op:
        batch_op.add_column(sa.Column("action_feedback", sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("reviews", schema=None) as batch_op:
        batch_op.drop_column("action_feedback")
    with op.batch_alter_table("events", schema=None) as batch_op:
        batch_op.drop_column("actions")
