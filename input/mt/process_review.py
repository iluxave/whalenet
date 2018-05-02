#!/usr/bin/python3

import csv
import json
import sys
import pprint

def usage():
    print("Usage: {nm} <batch.csv> <whales.json>".format(nm=sys.argv[0]))
    exit(1)

if len(sys.argv) != 3:
    usage()

hits_file=sys.argv[1]
review_file=sys.argv[2]
hitrev_file=hits_file+'.rev.csv'

def is_annotation_good(jpeg):
    """
    check if annotation was approved.
    True: approved
    False: Not yet reviewed
    None: Rejected
    """
    for a in annotations:
        if a['filename']=='train/'+jpeg:
            if not 'annotations' in a or len(a['annotations'])<1:
                return None
            return a['annotations'][0]['corrected']
    return False

with open(review_file, 'r') as rfile:
    annotations=json.load(rfile)

areader=csv.DictReader(open(hits_file, 'r'))
awriter=csv.DictWriter(open(hitrev_file, 'w'), areader.fieldnames)
awriter.writeheader()
for row in areader:
    good = is_annotation_good(row['Answer.imageid'])
    if good is None:
        row['Reject']='x'
    elif good == True:
        row['Approve']='x'
    else:
        continue
    awriter.writerow(row)
