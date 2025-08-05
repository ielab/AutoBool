# Entrez API Service

FastAPI service for querying PubMed via NCBI Entrez API with automatic date splitting, rate limiting, and concurrent processing.

## Features

- **Concurrent Processing**: Multiple API keys with automatic failover
- **Rate Limiting**: Intelligent cooldown management for API keys
- **Date Splitting**: Automatically splits large queries by date ranges
- **Resilient**: Retry logic with exponential backoff
- **Scalable**: Docker-based deployment with configurable workers

## Quick Start

### Option 1: Docker Compose (Recommended)
```bash
# Navigate to the API directory
cd train_autobool/entrez_api

# Build and start the service
docker-compose up --build -d

# Check logs
docker-compose logs -f

# Stop the service
docker-compose down
```

### Option 2: Direct Docker Commands
```bash
# Navigate to the API directory
cd train_autobool/entrez_api

# Build the image
docker build -t entrez-api .

# Run the container
docker run -d -p 8000:8000 --name entrez-api --restart unless-stopped entrez-api

# Check logs
docker logs -f entrez-api

# Stop and remove
docker stop entrez-api && docker rm entrez-api
```

### Option 3: Production Deployment
```bash
# For remote deployment, push to a registry:
docker build -t your-username/entrez-api:latest .
docker push your-username/entrez-api:latest

# Then on production server:
docker pull your-username/entrez-api:latest
docker run -d -p 8000:8000 --restart unless-stopped your-username/entrez-api:latest
```

## API Endpoints

### POST /entrez/query
Submit a PubMed query with date range filtering.

**Request Body:**
```json
{
  "query": "\"diabetes mellitus\"[MeSH Terms]",
  "mindate": "2020/01/01", 
  "maxdate": "2021/01/01"
}
```

**Response:**
```json
{
  "query": "\"diabetes mellitus\"[MeSH Terms]",
  "date_range": "2020/01/01 to 2021/01/01",
  "total_retrieved": 1250,
  "exceeded_overall_max": false,
  "ids": ["12345678", "87654321", ...],
  "errors": []
}
```

### GET /docs
Interactive API documentation (Swagger UI)

## Testing the API

```bash
# Health check
curl http://localhost:8000/docs

# Test query
curl -X POST "http://localhost:8000/entrez/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "\"diabetes mellitus\"[MeSH Terms]",
    "mindate": "2020/01/01", 
    "maxdate": "2021/01/01"
  }'
```

## Configuration

### API Keys Setup
Before deploying, you need to add your NCBI API keys to `entrez_submission_api.py`:

```python
API_KEYS = [
    {"email": "your_email@domain.com", "key": "your_api_key_here"},
    {"email": "backup_email@domain.com", "key": "backup_api_key_here"},
]
```

### Environment Variables
- `PYTHONUNBUFFERED=1`: Ensures real-time logging in Docker

### Configuration Parameters
- `RETMAX = 10_000`: Maximum results per API call
- `OVERALL_MAX = 200_000`: Maximum total results before stopping
- `CONCURRENT_REQUESTS = 4`: Number of concurrent workers
- `KEY_COOLDOWN_SECONDS = 3`: Cooldown time for failed API keys
- `MAX_RETRIES_PER_CHUNK = 5`: Retry attempts per date chunk

## How It Works

1. **Initial Count Check**: Queries total results to avoid exceeding limits
2. **Date Splitting**: Automatically splits large date ranges into smaller chunks
3. **Concurrent Processing**: Uses multiple workers with different API keys
4. **Rate Limiting**: Manages API key cooldowns to avoid 429 errors
5. **Resilient Retries**: Exponential backoff for failed requests

## Integration with Training

The training pipeline automatically connects to this API. Make sure the service is running before starting GRPO training:

```bash
# Start the API service
docker-compose up -d

# Then run your training
cd ../
python train_grpo.py --your-training-args
```

## Logs and Monitoring

```bash
# View real-time logs
docker-compose logs -f

# Check container status
docker-compose ps

# View resource usage
docker stats entrez-api
```