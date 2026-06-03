from sqlalchemy import Column, String, DateTime
from sqlalchemy.sql import func

from .database import Base


class Paper(Base):
    __tablename__ = "papers"

    paper_id = Column(String, primary_key=True)

    user_id = Column(
        String,
        nullable=False,
        index=True
    )

    filename = Column(
        String,
        nullable=False
    )

    uploaded_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )