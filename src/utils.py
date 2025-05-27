import os
import logging
from functools import lru_cache

import boto3
from botocore.config import Config as BotoConfig
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_logger(name: str = __name__, level: str = None) -> logging.Logger:
    """
    Configure and return a logger with standardized formatting and level.
    """
    logger = logging.getLogger(name)
    log_level = level or os.getenv("LOG_LEVEL", "INFO")
    logger.setLevel(log_level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


@lru_cache()
def get_aws_client(service_name: str, region: str = None):
    """
    Returns a boto3 client for the specified AWS service, with retry configuration.
    """
    session = boto3.session.Session()
    region = region or os.getenv("AWS_REGION")
    config = BotoConfig(
        retries={"max_attempts": 5, "mode": "standard"}
    )
    return session.client(service_name, region_name=region, config=config)


def load_config() -> dict:
    """
    Load and return application configuration from environment variables.
    """
    return {
        "AWS_REGION": os.getenv("AWS_REGION"),
        "S3_BUCKET": os.getenv("S3_BUCKET"),
        "QDRANT_URL": os.getenv("QDRANT_URL"),
        "OPENSEARCH_HOST": os.getenv("OPENSEARCH_HOST"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL"),
    }
