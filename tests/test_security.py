from app.core.security import hash_password, verify_password, create_access_token, decode_access_token

def test_password_hash_roundtrip():
    encoded = hash_password("strong-password")
    assert encoded != "strong-password"
    assert verify_password("strong-password", encoded)
    assert not verify_password("wrong-password", encoded)

def test_access_token_roundtrip():
    token = create_access_token("42")
    payload = decode_access_token(token)
    assert payload["sub"] == "42"
