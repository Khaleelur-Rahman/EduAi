import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from typing import Generator, Optional

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./whatsapp_tutor.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String(20), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=True)
    age = Column(Integer, nullable=True)
    country = Column(String(100), nullable=True)
    preferred_subjects = Column(Text, nullable=True)
    learning_mode = Column(String(20), nullable=True)
    language = Column(String(10), default="en", nullable=False)
    
    onboarding_step = Column(String(50), default="name", nullable=False)
    is_onboarded = Column(Boolean, default=False, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    progress = relationship("Progress", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(phone_number={self.phone_number}, name={self.name}, onboarded={self.is_onboarded})>"


class Progress(Base):
    __tablename__ = "progress"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    topic = Column(String(200), nullable=False)
    lesson_content = Column(Text, nullable=False)
    lesson_step = Column(Integer, default=1, nullable=False)
    total_steps = Column(Integer, default=1, nullable=False)
    
    # RAG lesson tracking
    chunk_id = Column(String(100), nullable=True)  # Current chunk position for /next continuation
    is_rag_lesson = Column(Boolean, default=False, nullable=False)
    rag_context = Column(Text, nullable=True)
    
    completed = Column(Boolean, default=False, nullable=False)
    score = Column(Integer, nullable=True)
    
    hidden = Column(Boolean, default=False, nullable=False)  # User hid this entry from their dashboard
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    user = relationship("User", back_populates="progress")
    quizzes = relationship("QuizProgress", back_populates="lesson", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Progress(user_id={self.user_id}, topic={self.topic}, step={self.lesson_step}/{self.total_steps}, rag={self.is_rag_lesson})>"


class QuizProgress(Base):
    __tablename__ = "quiz_progress"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    lesson_id = Column(Integer, ForeignKey("progress.id"), nullable=False)
    
    topic = Column(String(200), nullable=False)
    lesson_step = Column(Integer, nullable=False)
    chunk_id = Column(String(100), nullable=False)
    
    questions = Column(Text, nullable=False)   # JSON array of question objects
    user_answers = Column(Text, nullable=True)  # Encoded answer string, e.g. "1A 2B 3True"
    score = Column(Integer, nullable=True)      # Number of correct answers
    
    completed = Column(Boolean, default=False, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    user = relationship("User")
    lesson = relationship("Progress", back_populates="quizzes")
    
    def __repr__(self):
        return f"<QuizProgress(user_id={self.user_id}, topic={self.topic}, step={self.lesson_step}, completed={self.completed})>"


class DashboardCode(Base):
    """One-time codes for dashboard PIN auth (sent via WhatsApp)."""
    __tablename__ = "dashboard_codes"

    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String(20), nullable=False, index=True)
    code_hash = Column(String(64), nullable=False)  # SHA-256 hex digest of the 6-digit code
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<DashboardCode(phone_number={self.phone_number}, expires_at={self.expires_at})>"


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    Base.metadata.create_all(bind=engine)
    # Migrate: add `hidden` column to existing databases (no-op if already present)
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            if "sqlite" in DATABASE_URL:
                conn.execute(text("ALTER TABLE progress ADD COLUMN hidden INTEGER DEFAULT 0"))
            else:
                conn.execute(text("ALTER TABLE progress ADD COLUMN hidden BOOLEAN DEFAULT false"))
            conn.commit()
    except Exception:
        pass


def get_user_by_phone(db: Session, phone_number: str) -> User:
    return db.query(User).filter(User.phone_number == phone_number).first()


def create_user(db: Session, phone_number: str) -> User:
    user = User(phone_number=phone_number)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def update_user(db: Session, user: User, **kwargs) -> User:
    for key, value in kwargs.items():
        if hasattr(user, key):
            setattr(user, key, value)
    
    user.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(user)
    return user


def get_user_progress(db: Session, user_id: int, limit: int = 10, include_hidden: bool = False):
    q = db.query(Progress).filter(Progress.user_id == user_id)
    if not include_hidden:
        q = q.filter(Progress.hidden == False)
    return q.order_by(Progress.created_at.desc()).limit(limit).all()


def set_progress_hidden(db: Session, progress_id: int, user_id: int) -> bool:
    """Set progress row as hidden (remove from dashboard). Returns True if updated."""
    row = db.query(Progress).filter(Progress.id == progress_id, Progress.user_id == user_id).first()
    if not row:
        return False
    row.hidden = True
    row.updated_at = datetime.utcnow()
    db.commit()
    return True


def get_completed_lessons(db: Session, user_id: int, limit: int = 10):
    """Return lessons marked as completed, ordered by most recent first."""
    return db.query(Progress).filter(
        Progress.user_id == user_id,
        Progress.completed == True
    ).order_by(Progress.updated_at.desc()).limit(limit).all()


def create_progress(db: Session, user_id: int, topic: str, lesson_content: str, total_steps: int = 1, 
                   is_rag_lesson: bool = False, chunk_id: str = None) -> Progress:
    progress = Progress(
        user_id=user_id,
        topic=topic,
        lesson_content=lesson_content,
        total_steps=total_steps,
        is_rag_lesson=is_rag_lesson,
        chunk_id=chunk_id
    )
    db.add(progress)
    db.commit()
    db.refresh(progress)
    return progress


def get_current_lesson(db: Session, user_id: int) -> Progress:
    return db.query(Progress).filter(
        Progress.user_id == user_id,
        Progress.completed == False
    ).order_by(Progress.created_at.desc()).first()


def update_progress(db: Session, progress: Progress, **kwargs) -> Progress:
    for field, value in kwargs.items():
        if hasattr(progress, field):
            setattr(progress, field, value)
    
    progress.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(progress)
    return progress


def create_quiz_progress(db: Session, user_id: int, lesson_id: int, topic: str, 
                        lesson_step: int, chunk_id: str, questions: str) -> QuizProgress:
    quiz = QuizProgress(
        user_id=user_id,
        lesson_id=lesson_id,
        topic=topic,
        lesson_step=lesson_step,
        chunk_id=chunk_id,
        questions=questions
    )
    db.add(quiz)
    db.commit()
    db.refresh(quiz)
    return quiz


def get_current_quiz(db: Session, user_id: int) -> QuizProgress:
    return db.query(QuizProgress).filter(
        QuizProgress.user_id == user_id,
        QuizProgress.completed == False
    ).order_by(QuizProgress.created_at.desc()).first()


def update_quiz_progress(db: Session, quiz: QuizProgress, **kwargs) -> QuizProgress:
    for field, value in kwargs.items():
        if hasattr(quiz, field):
            setattr(quiz, field, value)
    
    quiz.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(quiz)
    return quiz


def get_user_quizzes(db: Session, user_id: int, limit: int = 10):
    return db.query(QuizProgress).filter(QuizProgress.user_id == user_id).order_by(QuizProgress.created_at.desc()).limit(limit).all()


def get_completed_quizzes(db: Session, user_id: int, limit: int = 10):
    """Return completed quizzes with scores, ordered by most recent first."""
    return db.query(QuizProgress).filter(
        QuizProgress.user_id == user_id,
        QuizProgress.completed == True
    ).order_by(QuizProgress.updated_at.desc()).limit(limit).all()


def get_dashboard_code(db: Session, phone_number: str) -> Optional[DashboardCode]:
    """Get the latest dashboard code for this phone that has not expired."""
    return db.query(DashboardCode).filter(
        DashboardCode.phone_number == phone_number,
        DashboardCode.expires_at > datetime.utcnow()
    ).order_by(DashboardCode.created_at.desc()).first()


def create_dashboard_code(db: Session, phone_number: str, code_hash: str, expires_at: datetime) -> DashboardCode:
    """Store a new dashboard code (replace any existing for this phone)."""
    db.query(DashboardCode).filter(DashboardCode.phone_number == phone_number).delete()
    row = DashboardCode(phone_number=phone_number, code_hash=code_hash, expires_at=expires_at)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def delete_dashboard_code(db: Session, phone_number: str) -> None:
    """Remove dashboard code(s) for this phone after successful verification."""
    db.query(DashboardCode).filter(DashboardCode.phone_number == phone_number).delete()
    db.commit()


if __name__ == "__main__":
    create_tables()
    print("Database tables created successfully!")
