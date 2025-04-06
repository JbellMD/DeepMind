#!/bin/bash

# Function to display help message
show_help() {
    echo "DeepMind Docker Helper Scripts"
    echo ""
    echo "Usage:"
    echo "  ./docker-scripts.sh [command]"
    echo ""
    echo "Commands:"
    echo "  start     - Start the Docker containers"
    echo "  stop      - Stop the Docker containers"
    echo "  restart   - Restart the Docker containers"
    echo "  build     - Rebuild the Docker containers"
    echo "  logs      - View Docker container logs"
    echo "  clean     - Clean up Docker containers and volumes"
    echo "  help      - Show this help message"
}

# Function to start containers
start() {
    echo "Starting Docker containers..."
    docker-compose up -d
}

# Function to stop containers
stop() {
    echo "Stopping Docker containers..."
    docker-compose down
}

# Function to restart containers
restart() {
    echo "Restarting Docker containers..."
    docker-compose restart
}

# Function to rebuild containers
build() {
    echo "Rebuilding Docker containers..."
    docker-compose build --no-cache
    docker-compose up -d
}

# Function to view logs
logs() {
    echo "Viewing Docker logs..."
    docker-compose logs -f
}

# Function to clean up
clean() {
    echo "Cleaning up Docker containers and volumes..."
    docker-compose down -v
    docker system prune -f
}

# Main script logic
case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    build)
        build
        ;;
    logs)
        logs
        ;;
    clean)
        clean
        ;;
    help|*)
        show_help
        ;;
esac
