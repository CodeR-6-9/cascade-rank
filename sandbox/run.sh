#!/usr/bin/env bash
# run.sh — Single command to produce submission.csv from candidates.jsonl
# Usage: bash run.sh
# Constraint: 5 min, 16 GB RAM, CPU only, no network

set -euo pipefail

CANDIDATES="data/raw/candidates.jsonl"
OUTPUT="data/output/final_submission.csv"

echo "=========================================="
echo "  Redrob Candidate Ranking Pipeline"
echo "=========================================="

# Check data file exists
if [ ! -f "$CANDIDATES" ]; then
    echo "ERROR: $CANDIDATES not found."
    echo "Copy the dataset first:"
    echo "  cp /path/to/candidates.jsonl data/raw/"
    exit 1
fi

# Create output dir
mkdir -p data/output

# Run pipeline
echo "Starting pipeline..."
python3 src/pipeline.py --output "$OUTPUT"

# Validate output
echo ""
echo "Validating submission..."
python3 validate_submission.py "$OUTPUT"

echo ""
echo "Done. Submission ready at: $OUTPUT"