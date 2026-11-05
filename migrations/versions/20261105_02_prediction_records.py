"""Add a queryable prediction record table.

Revision ID: 20261105_02
Revises: 20261105_01
"""
from alembic import op

from diabetes_demo.database import PredictionRecord

revision = "20261105_02"
down_revision = "20261105_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    PredictionRecord.__table__.create(bind=op.get_bind(), checkfirst=True)


def downgrade() -> None:
    PredictionRecord.__table__.drop(bind=op.get_bind(), checkfirst=True)
