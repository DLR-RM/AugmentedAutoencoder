#!/usr/bin/env bash
export AE_WORKSPACE_PATH="$(pwd)/AAE_workspace"
echo $AE_WORKSPACE_PATH
cd $AE_WORKSPACE_PATH;ae_init_workspace;cd ..
