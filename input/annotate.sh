#!/bin/bash
MYDIR=$(cd $(dirname $0); pwd)
export PYTHONPATH=$MYDIR/../sloth/sloth:$PYTHONPATH
export PATH=$MYDIR/../sloth/sloth/bin:$PATH
sloth --config whale_sloth.py whales.json
