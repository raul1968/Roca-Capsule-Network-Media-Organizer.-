# production/docker-compose.yml
version: '3.8'

services:
  # Main registry service
  roca-registry:
    build: .
    ports:
      - "8765:8765"  # Web portal
      - "8766:8766"  # REST API
    volumes:
      - ./data:/data
      - ./config:/config
      - ./logs:/logs
    environment:
      - REGISTRY_MODE=enterprise
      - DATABASE_PATH=/data/registry.db
      - CACHE_SIZE=1024
      - MAX_WORKERS=32
    restart: unless-stopped
    networks:
      - roca-network

  # Database (optional, for enterprise)
  postgres:
    image: postgres:14
    environment:
      - POSTGRES_DB=roca_registry
      - POSTGRES_USER=roca
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - roca-network

  # Redis cache
  redis:
    image: redis:7-alpine
    command: redis-server --maxmemory 1gb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    networks:
      - roca-network

  # Web portal with Nginx
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - roca-registry
    networks:
      - roca-network

volumes:
  postgres_data:
  redis_data:

networks:
  roca-network:
    driver: bridge