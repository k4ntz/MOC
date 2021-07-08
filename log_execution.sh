#!/bin/sh
set -e
set -x
description=$1
echo "$(date)" "|" "${description}" "|" "$@" >> experiment_log
shift
"$@"
