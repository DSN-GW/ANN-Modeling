#!/bin/bash
set -e

FILE_ID="1zPUKv2Lfs3nRnSDbmLSGHfzvOA_qSamo"
OUTPUT_NAME="data.zip"

# Google Drive direct download
FILE_URL="https://drive.google.com/uc?export=download&id=${FILE_ID}"

echo "Downloading using curl..."
curl -L "$FILE_URL" -o "$OUTPUT_NAME"

echo "Saved as $OUTPUT_NAME"

DIR_NAME=$(unzip -Z1 "$OUTPUT_NAME" | head -1 | cut -d'/' -f1)

if [ -d "$DIR_NAME" ]; then
    rm -rf "$DIR_NAME"
fi

unzip -o "$OUTPUT_NAME"
