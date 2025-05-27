import os
import pandas as pd
import pyarrow.parquet as pq
import pytest

from pathlib import Path
from src.data_ingestion import (
    extract_text_from_csv,
    chunk_text,
    write_parquet,
    upload_to_s3,
)


def test_extract_text_from_csv_with_column(tmp_path):
    # Prepare a CSV with a specific text column
    data = {"col1": ["hello", "world"], "col2": ["foo", "bar"]}
    csv_path = tmp_path / "test.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)

    text = extract_text_from_csv(csv_path, text_column="col1")
    assert text == "hello\nworld"


def test_extract_text_from_csv_without_column(tmp_path):
    # Prepare a CSV without specifying text_column
    data = {"a": ["x", "y"], "b": ["z", "w"]}
    csv_path = tmp_path / "test2.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)

    text = extract_text_from_csv(csv_path)
    # Should contain all text values
    for val in ["x", "y", "z", "w"]:
        assert val in text


def test_chunk_text_small():
    text = "a" * 25
    chunks = chunk_text(text, max_chars=10, overlap=2)
    # Expect ceil((25 - 10) / (10 - 2)) + 1 = ceil(15/8)+1 = 2+1 = 3 chunks
    assert len(chunks) == 3
    # Each chunk length <= max_chars
    assert all(len(c) <= 10 for c in chunks)


def test_write_and_read_parquet(tmp_path):
    chunks = ["foo", "bar", "baz"]
    out_file = tmp_path / "chunks.parquet"
    write_parquet(chunks, out_file)

    # Read back to confirm
    table = pq.read_table(str(out_file))
    df = table.to_pandas()
    assert list(df["text"]) == chunks
    assert list(df["id"]) == [0, 1, 2]


def test_upload_to_s3(monkeypatch, tmp_path):
    # Create a dummy file to upload
    local_file = tmp_path / "dummy.txt"
    local_file.write_text("content")

    calls = {}

    class DummyClient:
        def upload_file(self, Filename, Bucket, Key):
            calls['Filename'] = Filename
            calls['Bucket'] = Bucket
            calls['Key'] = Key

    # Monkeypatch boto3 session client
    monkeypatch.setattr(
        'boto3.session.Session.client',
        lambda self, service, region_name=None: DummyClient()
    )

    upload_to_s3(local_file, bucket="test-bucket", key="pref/dummy.txt")

    assert calls['Bucket'] == "test-bucket"
    assert calls['Key'] == "pref/dummy.txt"
    assert calls['Filename'] == str(local_file)
