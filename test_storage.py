"""Test storage module thoroughly."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.storage import StorageBackend, LocalStorage, MinIOStorage, get_storage

errors = []

# Test 1: Imports
print("1. Imports: OK")

# Test 2: LocalStorage init
try:
    ls = LocalStorage("./storage")
    print(f"2. LocalStorage init: OK (base_dir={ls.base_dir})")
except Exception as e:
    errors.append(f"LocalStorage init: {e}")
    print(f"2. LocalStorage init: FAIL - {e}")

# Test 3: get_url
url = ls.get_url("uploads/test.jpg")
assert url == "/files/uploads/test.jpg", f"Expected /files/uploads/test.jpg, got {url}"
print(f"3. LocalStorage.get_url: OK -> {url}")

# Test 4: get_full_path
fp = ls.get_full_path("uploads/test.jpg")
print(f"4. LocalStorage.get_full_path: OK -> {fp}")

# Test 5: save and delete
with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
    f.write("test content")
    tmp_path = f.name
try:
    result = ls.save(tmp_path, "test/test_file.txt")
    print(f"5. LocalStorage.save: OK -> {result}")
    assert ls.get_full_path("test/test_file.txt").exists(), "Saved file doesn't exist!"
    print("6. File exists after save: OK")
    deleted = ls.delete("test/test_file.txt")
    assert deleted, "Delete returned False"
    assert not ls.get_full_path("test/test_file.txt").exists(), "File still exists after delete!"
    print("7. LocalStorage.delete: OK")
finally:
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)

# Test 8: get_storage factory
os.environ.pop("STORAGE_TYPE", None)
s = get_storage()
assert isinstance(s, LocalStorage), f"Expected LocalStorage, got {type(s).__name__}"
print(f"8. get_storage() default: OK -> {type(s).__name__}")

# Test 9: MinIOStorage class structure
for method in ["save", "get_url", "delete", "get_full_path", "get_object"]:
    assert hasattr(MinIOStorage, method), f"MinIOStorage missing method: {method}"
print("9. MinIOStorage methods: OK (save, get_url, delete, get_full_path, get_object)")

# Test 10: MinIOStorage won't crash on import (just don't connect)
print("10. MinIOStorage class definition: OK")

# Test 11: Check MinIO constructor catches connection errors
os.environ["STORAGE_TYPE"] = "minio"
os.environ["MINIO_ENDPOINT"] = "localhost:59999"  # non-existent
try:
    ms = get_storage()
    print("11. MinIOStorage with bad endpoint: OK (didn't crash)")
except Exception as e:
    errors.append(f"MinIOStorage init crash: {e}")
    print(f"11. MinIOStorage with bad endpoint: FAIL - {e}")
finally:
    os.environ["STORAGE_TYPE"] = "local"

if errors:
    print(f"\n*** {len(errors)} ERROR(S) FOUND ***")
    for e in errors:
        print(f"  - {e}")
else:
    print("\n*** ALL TESTS PASSED ***")
