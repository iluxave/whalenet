#!/usr/bin/python3

import json
import sys
import pprint
from os.path import isfile

def usage():
    print("Usage: {nm} <out.json> <in1.json in2.json ...>".format(nm=sys.argv[0]))
    exit(1)

if len(sys.argv) < 3:
    usage()

out_file=sys.argv[1]

infiles=[]
i=2
while i<len(sys.argv):
    if not isfile(sys.argv[i]):
        print("File", sys.argv[i], "does not exist")
        usage()
    infiles.append(sys.argv[i])
    i+=1

out=open(out_file, 'w')
out_annotations=[]
for inf in infiles:
    f = open(inf)
    in_annotations=json.load(f)
    for a in in_annotations:
        if 'annotations' in a and len(a['annotations'])>0 and a['annotations'][0]['corrected']:
            out_annotations.append(a)
    f.close()

json.dump(out_annotations, out)
out.close()

            
