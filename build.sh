#!/bin/bash
unset PYTHONPATH

SCRIPT_DIR=`pwd`
INSTALLFOLDER=${SCRIPT_DIR}/install/${DLRRM_HOST_PLATFORM}
export PYTHONUSERBASE=${INSTALLFOLDER}

/usr/bin/pip2 install --upgrade --user .