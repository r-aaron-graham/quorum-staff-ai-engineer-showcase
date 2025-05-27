# Quorum Staff AI Engineer Showcase

A hands-on demonstration of GenAI pipelines, RAG architectures, and production-grade AI services—built to mirror the core responsibilities and technologies described in Quorum’s Staff AI Engineer role.

---

## 📖 Table of Contents

1. [Overview](#overview)  
2. [Key Features & Skill Mapping](#key-features--skill-mapping)  
3. [Architecture](#architecture)  
4. [Prerequisites](#prerequisites)  
5. [Installation](#installation)  
6. [Usage](#usage)  
7. [Testing & Evaluation](#testing--evaluation)  
8. [Infrastructure as Code](#infrastructure-as-code)  
9. [Contributing](#contributing)  
10. [License](#license)

---

## Overview

Quorum is on the cutting edge of AI-powered public affairs. This showcase repo contains:

- **Data Ingestion**: Batch and streaming ingestion of legislative texts, policy PDFs, and social media feeds.  
- **Vector Store Abstraction**: Plug-and-play wrappers for Qdrant or OpenSearch.  
- **RAG Pipeline**: Retrieval-Augmented Generation service built with LangChain.  
- **Semantic Search API**: FastAPI service exposing search & summarization endpoints.  
- **Evaluation Framework**: Precision/Recall metrics, feedback loops, and automated monitoring.  
- **Cloud-Native Deployment**: Docker, Terraform, and AWS Lambda/SageMaker examples.  

Each component is crafted to demonstrate the deep technical expertise, leadership mindset, and pragmatic engineering practices Quorum values.

---

## Key Features & Skill Mapping

| Component              | Quorum Skill & Responsibility                                          |
|------------------------|--------------------------------------------------------------------------|
| **data_ingestion.py**  | Modern data engineering: S3, streaming, ETL pipelines                    |
| **vector_store.py**    | Vector databases, semantic search (Qdrant/OpenSearch)                    |
| **rag_pipeline.py**    | RAG architectures, embeddings, LLM integration (LangChain/LlamaIndex)    |
| **sem_search.py**      | Production API: FastAPI/Django; performance tuning & latency optimization |
| **eval_metrics.py**    | LLM evaluation metrics; precision, recall, feedback-driven iteration     |
| **infra/terraform/**   | Scalable cloud infra: AWS (Lambda, DynamoDB, S3, IAM) via Terraform      |
| **tests/**             | Unit & integration tests; CI/CD best practices                           |

---

## Architecture

```text
  +------------+      +-------------+      +------------+
  | Raw Source | ---> | Ingestion   | ---> | S3 Bucket  |
  |  (PDF,     |      | (Airflow/   |      | (Parquet)  |
  |   API, RSS)|      |  Lambda)    |      +------------+
  +------------+             |                    |
                             v                    v
                       +-------------+      +------------+
                       | Embeddings  | <--- | Vector DB  |
                       | (AWS Sage-  |      +------------+
                       | Maker + Lang)|            |
                       +-------------+             |
                             |                     |
                             v                     |
                      +--------------+             |
                      | RAG Pipeline |-------------+
                      | (LangChain)  |
                      +--------------+
                             |
                             v
                       +-----------+
                       | FastAPI   |
                       | Service   |
                       +-----------+
