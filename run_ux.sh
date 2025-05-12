#!/bin/bash

DATA_FOLDER="data_small"
QUERIES_FOLDER="queries_small"
OUTPUT_FILE="./time.txt"
EXECUTABLE="./build/PC-project"

rm -f "$OUTPUT_FILE"

for query_file in "$QUERIES_FOLDER"/*; do
    query_name=$(basename "$query_file")

    start_time=$(date +%s%3N)

    "$EXECUTABLE" "$DATA_FOLDER" "$query_file"

    end_time=$(date +%s%3N)

    elapsed_time=$((end_time - start_time))

    echo "$query_name: $elapsed_time ms" >> "$OUTPUT_FILE"

    echo "Processed $query_name in $elapsed_time ms"
done

echo "All Done."

