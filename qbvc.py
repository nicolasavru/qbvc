#!/usr/bin/python2

import sys
import csv

from sig import *

def search(database, queryclip):
    """
    Returns a list of database rows which match queryclip.
    """
    sig = GenerateSignature(queryclip)
    sig_combined = CombineSignature(sig[1],sig[2])

    dbfile = open(database, 'r')
    reader = csv.reader(dbfile, delimiter=';')
    for row in reader:
        if CompareSignature(sig_combined, row[0]):
            result.append(row)

    dbfile.close()
    return result

def add(database, queryclip):
    """
    Adds signature of queryclip to database.
    """

    dbfile = open(database, 'a')
    sig = GenerateSignature(queryclip)
    sig_combined = CombineSignature(sig[1],sig[2])
    writer = csv.writer(dbfile, delimiter=";")
    writer.writerow([sig_combined, queryclip])
    dbfile.close()



commands = {
    "search" : search,
    "add" : add,
    }

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: ./gbvc.py database queryclip command")
        print("command is one of: search, add")
        sys.exit(-1)

    database = sys.argv[1]
    queryclip = sys.argv[2]
    command = sys.argv[3]

    commands[command](database, queryclip)
