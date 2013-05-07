#!/usr/bin/python2

import sys
import csv
import matplotlib.pyplot as plt

from sig import *

csv.field_size_limit(sys.maxsize)

def search(database, queryclip):
    """
    Returns a list of database rows which match queryclip.
    """
    sig = GenerateSignature(queryclip, demo)
    sig_combined = CombineSignature(sig[1],sig[2])

    dbfile = open(database, 'r')
    reader = csv.reader(dbfile, delimiter=';')
    result = []
    for row in reader:
        if CompareSignature(sig_combined, row[0], demo=demo):
            result.append(row)

    dbfile.close()
    return result

def add(database, queryclip):
    """
    Adds signature of queryclip to database.
    """

    dbfile = open(database, 'a')
    sig = GenerateSignature(queryclip, demo)
    sig_combined = CombineSignature(sig[1],sig[2])
    writer = csv.writer(dbfile, delimiter=";")
    writer.writerow([sig_combined, queryclip])
    dbfile.close()

demo = False

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
    if len(sys.argv) == 5 and sys.argv[4] == "demo":
        demo = True

    commands[command](database, queryclip)
    if demo:
        plt.show()
