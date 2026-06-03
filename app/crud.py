from .database import SessionLocal
from .models import Paper


def create_paper(
    paper_id: str,
    user_id: str,
    filename: str
):
    db = SessionLocal()

    try:
        paper = Paper(
            paper_id=paper_id,
            user_id=user_id,
            filename=filename
        )

        db.add(paper)
        db.commit()

    finally:
        db.close()