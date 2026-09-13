"""add fts column for hybrid search

Revision ID: a1f2c3d4e5b6
Revises: 3de3e54c74d0
Create Date: 2026-09-13 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'a1f2c3d4e5b6'
down_revision: Union[str, Sequence[str], None] = '3de3e54c74d0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("""
        ALTER TABLE chunk ADD COLUMN fts tsvector
        GENERATED ALWAYS AS (to_tsvector('english'::regconfig, content)) STORED
    """)
    op.execute("CREATE INDEX chunk_fts_idx ON chunk USING GIN (fts)")


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP INDEX IF EXISTS chunk_fts_idx")
    op.execute("ALTER TABLE chunk DROP COLUMN fts")
