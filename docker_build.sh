#!/bin/bash

echo "=== Building Custom EvalPlus Docker Image ==="
echo "This will create evalplus:latest"
echo ""

# Check if Dockerfile exists
if [ ! -f "Dockerfile" ]; then
    echo "Error: Dockerfile not found!"
    echo "Make sure you're running this script from the evalplus directory"
    exit 1
fi

echo "Building Docker image evalplus:latest..."
docker build -t evalplus:latest .

# Check if build was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=== Docker Image Built Successfully ==="
    echo "Image name: evalplus:latest"
else
    echo ""
    echo "=== Docker Build Failed ==="
    echo "Check the error messages above for details"
    exit 1
fi
