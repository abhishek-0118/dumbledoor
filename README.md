//local manual
python -m app.cli --index --env local

//prod manual
python -m app.cli --env prod --index

//serve
python -m app.cli --serve

Run in detached mode:
docker compose up -d                      # Local
docker compose --profile prod up -d       # Production

Rebuild images:
docker compose up --build                 # Local
docker compose --profile prod up --build  # Production

View logs:
docker compose logs -f                    # Local
docker compose --profile prod logs -f     # Production

Stop services:
docker compose down                       # Local
docker compose --profile prod down        # Production