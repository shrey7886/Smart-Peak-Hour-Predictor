# Smart Peak Hour Predictor

A machine learning application that predicts peak business hours using Temporal Fusion Transformer (TFT) models, with a REST API backend and Streamlit frontend.

## 🚀 Quick Start with Docker

### Prerequisites
- Docker Desktop installed and running
- At least 4GB of available RAM

### 1. Build the Docker Images
```bash
docker compose build
```

### 2. Start the Application
```bash
docker compose up -d
```

### 3. Access the Application
- **Streamlit Frontend**: http://localhost:8501
- **REST API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### 4. Stop the Application
```bash
docker compose down
```

## 🐳 Docker Commands Reference

### Build Images
```bash
# Build all services
docker compose build

# Build specific service
docker compose build api
docker compose build streamlit
```

### Run Services
```bash
# Start all services in background
docker compose up -d

# Start specific service
docker compose up -d api
docker compose up -d streamlit

# Start with logs visible
docker compose up
```

### Manage Services
```bash
# Stop all services
docker compose down

# Restart all services
docker compose restart

# View logs
docker compose logs
docker compose logs streamlit
docker compose logs api

# View running containers
docker ps
```

### Troubleshooting
```bash
# Check container status
docker ps -a

# Remove all containers and start fresh
docker compose down
docker system prune -f
docker compose up -d

# Access container shell
docker exec -it smart-peak-hour-streamlit bash
docker exec -it smart-peak-hour-api bash
```

## 📁 Project Structure

```
Smart-Peak-Hour-Predictor/
├── app.py                 # Streamlit frontend
├── graphql_api.py         # REST API backend
├── Dockerfile             # Container configuration
├── docker-compose.yml     # Multi-service orchestration
├── requirements.txt       # Python dependencies
├── data/                  # Data files
├── models/                # Trained ML models
└── utils/                 # Utility functions
```

## 🔧 Architecture

- **Frontend**: Streamlit web application
- **Backend**: FastAPI REST API
- **ML Model**: Temporal Fusion Transformer (TFT)
- **Containerization**: Docker with custom network
- **Communication**: HTTP requests between containers

## 📊 Features

- Peak hour prediction using TFT models
- Weather and holiday impact analysis
- Interactive data visualization
- REST API for predictions
- Containerized deployment
- Scalable microservices architecture

## 🛠️ Development

### Local Development (without Docker)
```bash
# Install dependencies
pip install -r requirements.txt

# Start REST API
uvicorn graphql_api:app --host 0.0.0.0 --port 8000

# Start Streamlit (in another terminal)
streamlit run app.py --server.port 8501
```

### Docker Development
```bash
# Rebuild after code changes
docker compose build
docker compose up -d

# View real-time logs
docker compose logs -f
```

## 🔍 Monitoring

### Check Service Health
```bash
# REST API health
curl http://localhost:8000/docs

# Streamlit health
curl http://localhost:8501
```

### View Container Logs
```bash
# All services
docker compose logs

# Specific service
docker compose logs streamlit
docker compose logs api
```

## 🚀 Production Deployment

### Environment Variables
Set these in your deployment environment:
- `MODEL_PATH`: Path to trained model file
- `HISTORICAL_DATA_PATH`: Path to historical data
- `API_URL`: REST API endpoint URL

### Scaling
```bash
# Scale Streamlit instances
docker compose up -d --scale streamlit=3

# Scale API instances
docker compose up -d --scale api=2
```

## 📝 API Documentation

The REST API provides the following endpoints:

- `POST /predict`: Generate predictions for future data
- `GET /docs`: Interactive API documentation (Swagger UI)

### Example API Request
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "timestamp": "2024-06-01T00:00:00",
      "hour": 0,
      "day_of_week": 6,
      "is_weekend": 1,
      "staff_count": 5,
      "promotion_flag": 0,
      "promotion_type": "None",
      "event_flag": 0,
      "event_name": "None",
      "inventory_alert": 0,
      "temp": 25.0,
      "humidity": 60.0,
      "rain": 0.0,
      "snow": 0.0,
      "wind_speed": 5.0,
      "clouds": 10,
      "is_holiday": 0,
      "holiday_type": "None",
      "holiday_name": "None",
      "weather_main": "Clear"
    }
  ]'
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with Docker
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.