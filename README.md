qbvc
====

(Query By Video Content) - video search by example

## Description ##

This project is based on [1], available at [2]. The shot-length, color-shift, and
centroid-based signatures are implemented, with minor modifications.

This project was created for ECE418 Digital Video at Cooper Union,
taught by Professor Fred Fontaine in the Spring 2013 term.

## Database Format ##

The database is a semicolon-delimited value file where each line has
the following format:

    signature;id

id is currently just the filename.

## Dependencies ##

* Python 2.7
* OpenCV
* Numpy
* Bottleneck

## Usage ##

### General Syntax ###

    python2 qbvc.py database\_file video\_file command [command_args] [demo]

`command` is one of:
* `add`
* `search`

A manditory `command_arg` for the `search` command is the score
function to use for signature comparison, one of:
* `linear`
* `cubic`
* `mean-weighted`

Appending "demo" as the last argument will print out some more numbers
and some nice color-shift histograms.

### Example ###
1. Create a database file and add a video file to it:
```
python2 qbvc.py database.db video1.mkv add
 ```

Or, add a directory tree of files:
```
    find . -name '*.mkv' -exec python2 qbvc.py database.db {} add \;
```

2. Search for the source of a video file in the database:
```
    python2 qbvc.py database.db video_clip.mkv search cubic
```

A good video clip to search for is a segment extracted from one of the
videos in the database. An easy way to extract a clip like that is with HandBrake [3]
For fun, try degrading the extracted segment by 


## Footnotes ##
[1]  Zobel, Justin, and Timothy C. Hoad. "Detection of video sequences
using compact signatures." ACM Transactions on Information Systems
(TOIS) 24.1 (2006): 1-50.

[2]  http://ww2.cs.mu.oz.au/~jz/fulltext/acmtois06.pdf

[3]  http://handbrake.fr/
