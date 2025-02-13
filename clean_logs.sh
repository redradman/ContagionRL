#!/bin/bash

# Print warning message
echo "Warning: This will delete all contents in the logs directory."
echo "Are you sure you want to continue? (y/n)"

# Read user input
read response

if [ "$response" = "y" ] || [ "$response" = "Y" ]; then
    # Check if logs directory exists
    if [ -d "logs" ]; then
        # Remove contents of logs directory
        rm -rf logs/*
        echo "Logs directory cleaned successfully!"
    else
        # Create logs directory if it doesn't exist
        mkdir logs
        echo "Logs directory created!"
    fi
else
    echo "Operation cancelled."
fi 