"""Create MySQL relational schema for datasets, models, predictions and agent traces."""
from alembic import op
from diabetes_demo.database import Base

revision = "20261105_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    Base.metadata.create_all(bind=op.get_bind(), checkfirst=True)


def downgrade() -> None:
    Base.metadata.drop_all(bind=op.get_bind(), checkfirst=True)
