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

echo "Running local tests"
#python $SCRIPT_DIR/local/test_local.py

echo "Starting Mixtera Server"
WORK_DIR=`mktemp -d -p "$DIR"`
if [[ ! "$WORK_DIR" || ! -d "$WORK_DIR" ]]; then
  echo "Could not create temp dir"
  exit 1
fi

function shutdown_server {
  echo "Shutting down server"
  if kill -0 "$server_pid" 2>/dev/null; then
    echo "Server is still running, killing it"
    # We need to get the PID of the Python server, which is a child of the bash script
    python_pid=$(pgrep -P "$server_pid")
    if [ -n "$python_pid" ]; then
      echo "Killing Python server process with PID $python_pid"
      pkill -15 -P "$python_pid"
      echo "Killed Python server"
    fi

    echo "Killing bash script"
    pkill -15 -P "$server_pid"
    echo "Killed it."

  fi
  echo "Server shut down."
}

function cleanup {
  echo "Exiting integration test script, running cleanup."
  rm -rf "$WORK_DIR"
  echo "Deleted temp working directory $WORK_DIR"
  
  shutdown_server

  if [ $script_exit_status -ne 0 ]; then
    echo "Tests did not run sucessfully, printing server output."
    echo "-- Server Output --"
    cat "$server_output"
    echo "-- Server Output --"
    echo "Finally exiting."
    exit $script_exit_status
  fi
}
trap cleanup EXIT

# TODO(#56): After the server has a better interface, there is no need to manually register the dataset before starting the server.
python $SCRIPT_DIR/prep_server.py $WORK_DIR

echo "Starting Mixtera server"

server_output=$(mktemp)
mixtera-server --port 6666 $WORK_DIR &> "$server_output" &
server_pid=$!

sleep 2

if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "Server crashed within 2 seconds after starting."
    cat "$server_output"
    exit 1
fi

echo "Server started."
script_exit_status=0

echo "Running server tests"
#python $SCRIPT_DIR/server/test_server.py || script_exit_status=$?

echo "Running mixtera torch dataset tests"
python $SCRIPT_DIR/mixtera_torch_dataset/test_torch_dataset.py || script_exit_status=$?

echo "Succesfully ran all integration tests."
