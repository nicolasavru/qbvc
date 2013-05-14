#!/usr/bin/python2

import sys
import csv
import heapq

from sig import *
import scorefuncs

csv.field_size_limit(sys.maxsize)

num_returned = 10

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
        if demo:
            print "Searching For Matches in", row[1]
        matches = CompareSignature(sig_combined, row[0],
                                   score_func=scorefunc,
                                   demo=demo, N=num_returned)
        if matches:
            cur_result = [(match[0], match[1], row[1]) for match in matches]
            result = list(heapq.merge(result,cur_result))

    dbfile.close()

    #print result[-num_returned:]
    print_results(result[-num_returned:])

    return result

def print_results(results, max_name_size=50):

    i = 1
    print "Rank\tVideo Name\t\t\t\t\t\tTime Into Video\t\tScore"
    for result in results[::-1]:
        vid_name = result[2]
        if len(vid_name) > max_name_size:
            vid_name = vid_name[0:max_name_size-3] + "..."
        else:
            vid_name += " " * (max_name_size-len(vid_name))
        s = "%d\t%s\t%i Seconds\t\t%d\n" % (i,vid_name,round(result[1]),result[0])
        print s
        i+=1

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

scorefuncdict = {
    "linear" : scorefuncs.LinearScoreFunction,
    "cubic" : scorefuncs.CubicScoreFunction,
    "mean-weighted" : scorefuncs.MeanWeightedScoreFunction
    }
scorefunc = scorefuncs.CubicScoreFunction

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: ./gbvc.py database queryclip command")
        print("command is one of: search, add")
        sys.exit(-1)

    database = sys.argv[1]
    queryclip = sys.argv[2]
    command = sys.argv[3]
    if command == "search":
        scorefunc = scorefuncdict[sys.argv[4]]
        if len(sys.argv) > 5:
            num_returned = int(sys.argv[5])

    if sys.argv[-1] == "demo":
        demo = True

    commands[command](database, queryclip)
