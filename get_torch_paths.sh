#!/bin/bash

INCLUDE_PATHS=`mkdir -p temp && pushd temp > /dev/null && python -c "from torch.utils.cpp_extension import include_paths; print(','.join(include_paths()))" && popd > /dev/null && rm -rf temp`
echo $INCLUDE_PATHS