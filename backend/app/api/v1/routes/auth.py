from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.core.security import create_access_token, decode_access_token, hash_password, verify_password
from app.db.session import get_db
from app.models.entities import User
from app.schemas.auth import RegisterRequest, LoginRequest

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/register")
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    email = str(request.email).lower()
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(status_code=409, detail="An account with that email already exists.")
    user = User(email=email, name=request.name, password_hash=hash_password(request.password))
    db.add(user); db.commit(); db.refresh(user)
    return {"id": user.id, "email": user.email, "name": user.name}

@router.post("/login")
def login(request: LoginRequest, db: Session = Depends(get_db)):
    email = str(request.email).lower()
    user = db.scalar(select(User).where(User.email == email))
    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    return {"access_token": create_access_token(str(user.id)), "token_type": "bearer"}

@router.get("/me")
def me(authorization: str | None = Header(default=None), db: Session = Depends(get_db)):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Bearer token required.")
    token = authorization.split(" ", 1)[1]
    try:
        payload = decode_access_token(token)
        user = db.get(User, int(payload["sub"]))
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired token.") from exc
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return {"id": user.id, "email": user.email, "name": user.name, "created_at": user.created_at}
