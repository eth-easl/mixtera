#!/bin/bash
set -e # stops execution on non zero exit code

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
echo "Integration tests are located in $SCRIPT_DIR"
echo "Running as user $USER"

echo "Checking whether Mixtera is available in current environment."
MIXTERAPATH="$(python -c 'import mixtera; print(mixtera.__path__[0])')"
if [ $? -eq 0 ]; then
  echo "Mixtera is available at $MIXTERAPATH"
else
  echo "Mixtera is not available. Please run this in an environment with Mixtera installed."
  exit 1
fi

echo "Running local data collection tests"
python $SCRIPT_DIR/local_data_collection/test_local_collection.py

echo "TODO: Start Mixtera Server here for tests" # TODO(MaxiBoether): do this.

echo "Running remote data collection tests"
python $SCRIPT_DIR/remote_data_collection/test_remote_collection.py

echo "Running mixtera torch dataset tests"
python $SCRIPT_DIR/mixtera_torch_dataset/test_torch_dataset.py

echo "Successfuly ran all integration tests."