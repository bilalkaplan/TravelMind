import sys
import os
import pytest

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cmu_rag_answer import stream_and_strip_think

class MockDelta:
    def __init__(self, content):
        self.content = content

class MockChoice:
    def __init__(self, content):
        self.delta = MockDelta(content)

class MockChunk:
    def __init__(self, content):
        self.choices = [MockChoice(content)]

def simulate_stream(chunks):
    """Convert a list of strings into mock API response chunks."""
    return [MockChunk(c) for c in chunks]

def consume_stream(chunks):
    """Consume the stream_and_strip_think generator and return results."""
    gen = stream_and_strip_think(simulate_stream(chunks), "en")
    return list(gen)

def test_single_chunk_think():
    chunks = ["<think>thinking</think>answer"]
    res = consume_stream(chunks)
    assert res == [
        {"type": "think", "content": "thinking"},
        {"type": "answer", "content": "answer"}
    ]

def test_split_think_tag():
    # Tag split across chunks
    chunks = ["<t", "hin", "k>my ", "thoughts</th", "ink>my answer"]
    res = consume_stream(chunks)
    
    # We expect some specific yields based on our buffering logic
    types = [r["type"] for r in res]
    assert "think" in types
    assert "answer" in types
    
    full_think = "".join(r["content"] for r in res if r["type"] == "think")
    full_answer = "".join(r["content"] for r in res if r["type"] == "answer")
    
    assert full_think == "my thoughts"
    assert full_answer == "my answer"

def test_no_think_tag():
    chunks = ["just ", "an ", "answer"]
    res = consume_stream(chunks)
    types = [r["type"] for r in res]
    
    assert "think" not in types
    full_answer = "".join(r["content"] for r in res if r["type"] == "answer")
    assert full_answer == "just an answer"

def test_unterminated_think():
    chunks = ["<think>still thinking"]
    res = consume_stream(chunks)
    
    types = [r["type"] for r in res]
    assert "think" in types
    assert "answer" not in types
    full_think = "".join(r["content"] for r in res if r["type"] == "think")
    assert full_think == "still thinking"

def test_false_alarm_tag():
    chunks = ["<this ", "is ", "math: 5 < 10"]
    res = consume_stream(chunks)
    
    full_answer = "".join(r["content"] for r in res if r["type"] == "answer")
    assert full_answer == "<this is math: 5 < 10"
