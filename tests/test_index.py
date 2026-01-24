from src.inverted_index import build_inverted_index

def test_inverted_index():
    index = build_inverted_index("data")

    assert "machine" in index
    assert "doc1.txt" in index["machine"]
