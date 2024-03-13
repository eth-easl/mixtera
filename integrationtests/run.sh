#!/bin/bash
set -e # stops execution on non zero exit code

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
echo "Integration tests are located in $SCRIPT_DIR"
echo "Running as user $USER"

echo "Running local data collection tests"
python $SCRIPT_DIR/test_local_collection.py

echo "Running remote data collection tests"
python $SCRIPT_DIR/test_remote_collection.py

echo "Running mixtera torch dataset tests"
python $SCRIPT_DIR/test_torch_dataset.py

echo "Successfuly ran all integration tests."