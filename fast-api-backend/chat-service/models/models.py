from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from database.connection import Base


class ChatSession(Base):
    """One conversation thread.

    Named ChatSession rather than Session because every module in this service also has
    SQLAlchemy's AsyncSession in scope, and two unrelated things called Session in one
    file is a bug waiting to be written.
    """
    __tablename__ = "chat_session"

    id: Mapped[int] = mapped_column(primary_key=True)
    # No ForeignKey: users live in auth-db, a different database on a different port, so
    # Postgres cannot enforce this one. Holds the JWT's `sub`, which auth-service sets to
    # str(user.id). A user deleted there leaves their sessions behind here — the same
    # stateless tradeoff /search already makes.
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class Interaction(Base):
    """One turn: the question and the answer it produced.

    A turn is stored as a single row rather than two message rows with a sender column.
    In RAG the pair is always generated together and never separately — there is no
    query without an answer — so splitting them buys a join and an impossible state
    (an answer with no question) in exchange for nothing.
    """
    __tablename__ = "interaction"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("chat_session.id", ondelete="CASCADE"), nullable=False, index=True
    )
    query: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    # Which wiki pages the answer was grounded in. The retrieved chunks themselves are not
    # kept, so this is the only way to audit a wrong answer after the fact.
    sources: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False)
    # Gemini's own id for this turn, handed back as previous_interaction_id on the next one
    # so the model keeps the thread without us replaying the transcript into every prompt.
    # Nullable because the provider may not return one.
    provider_interaction_id: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
