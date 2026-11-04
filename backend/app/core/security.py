import datetime as dt
from typing import Optional, Any, Dict
from passlib.context import CryptContext
from jose import jwt, JWTError
from fastapi import HTTPException
from app.core.config import settings

# Support argon2id hashes (created by PostgreSQL backend) and bcrypt (SQLite backend).
# New hashes are always bcrypt; argon2 entries are transparently verified.
pwd_context = CryptContext(schemes=["bcrypt", "argon2"], deprecated=[])

def validate_password_length(password: str) -> None:
    if len(password.encode("utf-8")) > 72:
        raise HTTPException(
            status_code=422,
            detail="Password exceeds the maximum supported length of 72 bytes."
        )

def verify_password(plain_password: str, hashed_password: str) -> bool:
    validate_password_length(plain_password)
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    validate_password_length(password)
    return pwd_context.hash(password)

def create_access_token(data: Dict[str, Any], expires_delta: Optional[dt.timedelta] = None) -> str:
    to_encode = data.copy()
    expire = dt.datetime.utcnow() + (expires_delta or dt.timedelta(minutes=settings.JWT_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.JWT_SECRET, algorithm="HS256")

def decode_access_token(token: str) -> Dict[str, Any]:
    try:
        return jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
    except JWTError as e:
        raise HTTPException(status_code=401, detail="Invalid or expired token") from e
