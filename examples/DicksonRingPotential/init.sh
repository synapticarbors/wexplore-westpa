#!/bin/bash

source env.sh

rm -f west.h5
BSTATES="--bstate initA,1.0"
$WEST_ROOT/bin/w_init $BSTATES "$@"

