provider "aws" {
  region = var.region
}

############################
# S3 Bucket for Ingestion
############################
resource "aws_s3_bucket" "ingestion_bucket" {
  bucket = var.s3_bucket_name
  acl    = "private"

  versioning {
    enabled = true
  }

  lifecycle_rule {
    id      = "expiration"
    enabled = true
    expiration {
      days = 90
    }
  }
}

############################
# DynamoDB Table for Metadata
############################
resource "aws_dynamodb_table" "metadata_table" {
  name           = var.dynamodb_table_name
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "document_id"
  attribute {
    name = "document_id"
    type = "S"
  }

  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }
}

############################
# IAM Role for Lambda
############################
resource "aws_iam_role" "lambda_exec_role" {
  name = "${var.lambda_name_prefix}-exec-role"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_exec_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "lambda_s3_dynamo" {
  name   = "lambda-s3-dynamo-policy"
  role   = aws_iam_role.lambda_exec_role.id
  policy = data.aws_iam_policy_document.lambda_s3_dynamo.json
}

data "aws_iam_policy_document" "lambda_s3_dynamo" {
  statement {
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "dynamodb:PutItem",
      "dynamodb:GetItem",
      "dynamodb:UpdateItem"
    ]
    resources = [
      aws_s3_bucket.ingestion_bucket.arn,
      "${aws_s3_bucket.ingestion_bucket.arn}/*",
      aws_dynamodb_table.metadata_table.arn,
      "${aws_dynamodb_table.metadata_table.arn}/*"
    ]
  }
}

############################
# AWS Lambda for Ingestion
############################
resource "aws_lambda_function" "ingestion" {
  function_name = "${var.lambda_name_prefix}-ingestion"
  role          = aws_iam_role.lambda_exec_role.arn
  handler       = "data_ingestion.main"
  runtime       = "python3.10"

  filename      = var.lambda_package_path
  source_code_hash = filebase64sha256(var.lambda_package_path)

  environment {
    variables = {
      S3_BUCKET = aws_s3_bucket.ingestion_bucket.bucket
      DYNAMODB_TABLE = aws_dynamodb_table.metadata_table.name
      AWS_REGION = var.region
    }
  }
}

############################
# SageMaker Endpoint for Embeddings
############################
resource "aws_sagemaker_model" "embedding_model" {
  name          = var.sagemaker_model_name
  execution_role_arn = aws_iam_role.lambda_exec_role.arn

  primary_container {
    image          = var.sagemaker_image_uri
    model_data_url = var.sagemaker_model_data_url
  }
}

resource "aws_sagemaker_endpoint_configuration" "embedding_cfg" {
  name = var.sagemaker_endpoint_config_name

  production_variants {
    variant_name          = "AllTraffic"
    model_name            = aws_sagemaker_model.embedding_model.name
    initial_instance_count = var.sagemaker_instance_count
    instance_type         = var.sagemaker_instance_type
  }
}

resource "aws_sagemaker_endpoint" "embedding_endpoint" {
  name = var.sagemaker_endpoint_name
  endpoint_config_name = aws_sagemaker_endpoint_configuration.embedding_cfg.name
}
