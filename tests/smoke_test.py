"""Smoke test against a running API. Does not start any services."""
import sys
import json
import urllib.request
import urllib.error

API_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
PASSED = 0
FAILED = 0


def check(name, fn):
    global PASSED, FAILED
    try:
        fn()
        print(f"  PASS  {name}")
        PASSED += 1
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        FAILED += 1


def test_health():
    res = urllib.request.urlopen(f"{API_URL}/health")
    data = json.loads(res.read())
    assert data["status"] == "healthy", f"unexpected status: {data}"


def test_predict():
    from PIL import Image
    import io
    img = Image.new("RGB", (224, 224), (128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    boundary = "----SmokeTestBoundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="test.jpg"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode() + img_bytes + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{API_URL}/predict",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    res = urllib.request.urlopen(req)
    data = json.loads(res.read())
    assert "label" in data, f"no label in response: {data}"
    assert "confidence" in data, f"no confidence in response: {data}"
    assert data["label"] in ("positive", "negative"), f"bad label: {data['label']}"


def test_history():
    res = urllib.request.urlopen(f"{API_URL}/history")
    data = json.loads(res.read())
    assert isinstance(data, list), f"history not a list: {type(data)}"


if __name__ == "__main__":
    print(f"Smoke testing {API_URL}")
    check("/health", test_health)
    check("/predict", test_predict)
    check("/history", test_history)
    print(f"\nResults: {PASSED} passed, {FAILED} failed")
    sys.exit(1 if FAILED > 0 else 0)

