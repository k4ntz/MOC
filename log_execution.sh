#!/bin/sh
set -e
set -x
git pull || true
describtion=$1
shift
$@
echo $(date) "|"${describtion}"|" $@ >> experiment_log
