#!/usr/bin/python3

import csv
import json
import sys
import pprint

def usage():
    print("Usage: {nm} <batch.csv>".format(nm=sys.argv[0]))
    exit(1)

if len(sys.argv) != 2:
    usage()

file=sys.argv[1]

annotations=[]
with open(file, 'r') as csvfile:
    areader=csv.DictReader(csvfile)
    for row in areader:
        if row['AssignmentStatus']=='Submitted':
            ann=json.loads(row['Answer.coordinates'])
            ann['worktime']=row['WorkTimeInSeconds']
            annotations.append(ann)

print(json.dumps(annotations))
