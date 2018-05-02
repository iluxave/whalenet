#!/bin/bash
MYDIR=$(cd $(dirname $0); pwd)
export PYTHONPATH=$MYDIR/../sloth/sloth:$PYTHONPATH
export PATH=$MYDIR/../sloth/sloth/bin:$PATH
#python -m pdb $MYDIR/../sloth/sloth/bin/sloth --config whale_sloth.py whales.json
WHALES=whales.json
if [ -f "$1" ]; then
	WHALES="$1"
fi

sloth --config whale_sloth.py $WHALES
