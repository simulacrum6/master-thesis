import csv
import sys

tsvlines = csv.reader(sys.stdin, dialect=csv.excel_tab)
csvlines = csv.writer(sys.stdout, dialect=csv.excel)
for row in tsvlines:
    csvlines.writerow(row)
