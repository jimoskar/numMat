#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
nice python3 $DIR/find_best_config.py testing_test_function1 
nice python3 

