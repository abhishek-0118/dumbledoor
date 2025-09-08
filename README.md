# Repository Indexing and Search Service

## Configuration
This service uses YAML configuration files:
- `config/local.yaml` - Local development environment
- `config/prod.yaml` - Production environment

## Setup Instructions

### Local Development Setup
1. **Copy Repository**: Copy the target repository folder into `repos/` directory
   ```bash
   cp -r /path/to/your/repo ./repos/
   ```

2. **Update Configuration**: Edit `config/local.yaml` and update the `indexing.local_paths` to point to your repository:
   ```yaml
   indexing:
     local_paths: ["./repos/your-repo-name"]
   ```

3. **Environment Variables**: Ensure your `.env` file contains required keys:
   ```bash
   GITHUB_TOKEN=your_github_token_here
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

### Production Setup
1. **Update Configuration**: Edit `config/prod.yaml` and configure `indexing.repo_urls`:
   ```yaml
   indexing:
     repo_urls:
       - 'https://github.com/your-org/your-repo.git'
   ```

2. **Environment Variables**: Set required environment variables for production.

## Command Line Usage

### Local Environment Commands
```bash
# Index repositories (local environment)
python -m app.cli --index --env local

# Force complete re-indexing (ignores previous state)
python -m app.cli --index --env local --force-reindex

# Clean all embeddings and start fresh
python -m app.cli --index --env local --force-cleanup

# Start API server (local)
python -m app.cli --serve --env local

# Test GitHub authentication
python -m app.cli --auth-test --env local

# List available repositories
python -m app.cli --list-repos --env local
```

### Production Environment Commands
```bash
# Index repositories (production environment)
python -m app.cli --index --env prod

# Force complete re-indexing (ignores previous state)
python -m app.cli --index --env prod --force-reindex

# Clean all embeddings and start fresh
python -m app.cli --index --env prod --force-cleanup

# Start API server (production)
python -m app.cli --serve --env prod

# Test GitHub authentication
python -m app.cli --auth-test --env prod

# List available repositories
python -m app.cli --list-repos --env prod
```

### Command Options
- `--index`: Index repositories according to configuration
- `--serve`: Start the FastAPI server
- `--env`: Override APP_ENV (local|prod)
- `--auth-test`: Test GitHub authentication
- `--list-repos`: List available repositories
- `--force-reindex`: Force complete re-indexing, ignoring previous state
- `--force-cleanup`: Clean all embeddings and vector store before indexing

## Docker Commands

### Local Development
```bash
# Run in detached mode
docker compose up -d

# Rebuild images
docker compose up --build

# View logs
docker compose logs -f

# Stop services
docker compose down
```

### Production
```bash
# Run in detached mode
docker compose --profile prod up -d

# Rebuild images
docker compose --profile prod up --build

# View logs
docker compose --profile prod logs -f

# Stop services
docker compose --profile prod down
```

## API Usage

Once the server is running, you can make requests to the `/ask` endpoint:

```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{
       "q": "what is the name of repo",
       "k": 12,
       "repo": "cds-web"
     }'
```

## Troubleshooting

### Force Cleanup When Needed
If you encounter issues with vector dimensions or corrupted embeddings:
```bash
python -m app.cli --force-cleanup --index --env local
```

### Repository Naming Issues
If search results show incorrect repository names, try force re-indexing:
```bash
python -m app.cli --index --env local --force-reindex
```