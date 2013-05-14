qbvc
====

(Query By Video Content) - video search by example

## Database Format ##

The database is a semicolon-delimited value file where each line has
the following format:

    signature;id

id is currently just the filename.

## Usage ##
python2 qbvc.py database_file video_file command [command_args] [demo]

command is one of:
add
search

A manditory command_arg for search is the score function for signature comparison, one of:
linear
cubic
mean-weighted

Currently the cubic scoring function is most effective.
