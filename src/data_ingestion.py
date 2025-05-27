import os
import argparse
import logging
from pathlib import Path
from typing import List, Dict

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import boto3

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None  # Optional dependency for PDF parsing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_text_from_pdf(path: Path) -> str:
    """
    Extracts text from a PDF file.
    """
    if PdfReader is None:
        raise ImportError("PyPDF2 is required for PDF parsing")

    reader = PdfReader(str(path))
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text)


def extract_text_from_csv(path: Path, text_column: str = None) -> str:
    """
    Reads a CSV/TSV and concatenates text from the specified column or all columns.
    """
    df = pd.read_csv(path)
    if text_column and text_column in df.columns:
        return "\n".join(df[text_column].astype(str).tolist())
    # Fallback: concatenate all string columns
    texts = []
    for col in df.select_dtypes(include=[object]).columns:
        texts.extend(df[col].astype(str).tolist())
    return "\n".join(texts)


def chunk_text(text: str, max_chars: int = 10000, overlap: int = 200) -> List[str]:
    """
    Splits text into chunks of max_chars with overlap for context.
    """
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def write_parquet(chunks: List[str], output_path: Path) -> None:
    """
    Writes a list of text chunks to a Parquet file with schema {id: int, text: str}.
    """
    df = pd.DataFrame({"id": list(range(len(chunks))), "text": chunks})
    table = pa.Table.from_pandas(df)
    pq.write_table(table, str(output_path))
    logger.info(f"Wrote {len(chunks)} chunks to {output_path}")


def upload_to_s3(local_path: Path, bucket: str, key: str, region: str = None) -> None:
    """
    Uploads a local file to S3.
    """
    session = boto3.session.Session()
    s3 = session.client("s3", region_name=region)
    s3.upload_file(str(local_path), bucket, key)
    logger.info(f"Uploaded {local_path} to s3://{bucket}/{key}")


def process_file(path: Path, output_dir: Path, bucket: str, prefix: str, max_chars: int, overlap: int, region: str = None) -> None:
    """
    Extracts text from a single file, chunks it, writes Parquet locally, and uploads to S3.
    """
    logger.info(f"Processing {path}")
    ext = path.suffix.lower()
    if ext == ".pdf":
        text = extract_text_from_pdf(path)
    elif ext in [".csv", ".tsv"]:
        text = extract_text_from_csv(path)
    else:
        text = path.read_text()

    chunks = chunk_text(text, max_chars=max_chars, overlap=overlap)

    local_file = output_dir / f"{path.stem}.parquet"
    write_parquet(chunks, local_file)

    s3_key = f"{prefix}/{local_file.name}"
    upload_to_s3(local_file, bucket, s3_key, region)


def main():
    parser = argparse.ArgumentParser(description="Ingest and preprocess legislative & policy documents.")
    parser.add_argument("--input", type=Path, required=True, help="Input directory or file path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Local directory for Parquet files.")
    parser.add_argument("--s3-bucket", type=str, required=True, help="Target S3 bucket name.")
    parser.add_argument("--s3-prefix", type=str, default="ingested", help="S3 key prefix.")
    parser.add_argument("--max-chars", type=int, default=10000, help="Max characters per chunk.")
    parser.add_argument("--overlap", type=int, default=200, help="Overlap characters between chunks.")
    parser.add_argument("--region", type=str, default=None, help="AWS region for S3.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = [args.input]
    if args.input.is_dir():
        paths = list(args.input.glob("**/*"))

    for path in paths:
        if path.is_file():
            process_file(path, args.output_dir, args.s3_bucket, args.s3_prefix, args.max_chars, args.overlap, args.region)

if __name__ == "__main__":
    main()
