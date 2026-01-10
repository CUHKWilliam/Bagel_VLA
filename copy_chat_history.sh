#!/bin/bash
# Script to copy chat history from remote server
# Usage: ./copy_chat_history.sh [username]

REMOTE_HOST="10.0.100.92"
REMOTE_PORT="56025"
REMOTE_PATH="/mnt/data/code/Bagel_VLA"
LOCAL_PATH="/mnt/data/code/Bagel_VLA"

# Use provided username or default to current user
USERNAME=${1:-$(whoami)}

echo "Attempting to copy chat history from ${USERNAME}@${REMOTE_HOST}:${REMOTE_PORT}..."
echo "Remote path: ${REMOTE_PATH}/.cursor"
echo "Local path: ${LOCAL_PATH}/"

# Try to copy .cursor directory (Cursor chat history)
echo "Copying .cursor directory..."
scp -P ${REMOTE_PORT} -r ${USERNAME}@${REMOTE_HOST}:${REMOTE_PATH}/.cursor ${LOCAL_PATH}/ 2>&1

# Also try to copy any chat-related files
echo "Checking for other chat history files..."
ssh -p ${REMOTE_PORT} ${USERNAME}@${REMOTE_HOST} "find ${REMOTE_PATH} -maxdepth 1 -name '*chat*' -o -name '*.cursor*' 2>/dev/null" | while read file; do
    if [ ! -z "$file" ]; then
        echo "Copying: $file"
        scp -P ${REMOTE_PORT} ${USERNAME}@${REMOTE_HOST}:${file} ${LOCAL_PATH}/
    fi
done

echo "Done!"

