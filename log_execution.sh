#!/bin/sh
set -e
set -x
description=$1
shift
$@
echo $(date) "|"${description}"|" $@ >> experiment_log
