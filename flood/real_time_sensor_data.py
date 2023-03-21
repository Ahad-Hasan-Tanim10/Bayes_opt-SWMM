#!/usr/bin/env python
"""
Download all data from a list of Adafruit IO feeds. Change constants in this
file in the CONFIGURATION section before you run the script.

Usage:

    $ START="2019-05-01T00:00Z" END="2019-06-01T00:00Z" \
            python download_paged_data.py
"""

from io import StringIO
import http.client
import json
import os
import re
import sys
import time
import urllib.parse

#########################
# CONFIGURATION
#########################


# ISO8601 formatted date strings:
#   for example, "2019-05-01T00:00:00Z" is May 1, 2019, midnight UTC
START_TIME = None
END_TIME = None

# replace this with the location in which you would like to store the data
DESTINATION = "/local/folder/path"

#   (label, Adafruit IO API feed key, file name )
# add a row for every feed you want to download
FEEDS = (("Counter 1", "example.counter-1", "counter-1.csv"),)

# replace with your Adafruit IO key
AIO_KEY = None

# replace with your Adafruit IO username
USERNAME = None


#########################
# END CONFIGURATION
#########################


def short_csv(record):
    """
    Choose how you want records to be stored. This function will be called
    once for every record in the resulting data set. Return None or False
    skip a record. The return value will be written to the final file with
    a newline "\n" character added.
    """
    import csv

    # guaranteed properly escaped CSV rows
    row = StringIO()
    writer = csv.writer(row)
    writer.writerow([record['created_epoch'], record['value']])
    return row.getvalue().strip()

def short_json(record):
    """
    Replace full IO JSON with truncated JSON records.
    """
    # properly generated JSON rows with "t" (created at timestamp) and "v" (value) keys
    return json.dumps({"t": record["created_at"], "v": record["value"]})

TRANSFORM = None


if sys.version_info < (3, 0):
    print("make sure you're using python3 or python version 3.0 or higher")
    exit(1)


def parse_next_value(instr):
    """
    Parse the `next` page URL in the pagination Link header.
    """
    if not instr:
        return None
    for link in [h.strip() for h in instr.split(";")]:
        if re.match('rel="next"', link):
            nurl_result = re.search("<(.+)>", link)
            if nurl_result:
                nurl = nurl_result[1]
                return nurl
    return None


def download(url, out_file, label, headers=None, transform=json.dumps):
    """
    Download a single chunk of data from Adafruit IO, write it to out_file
    (file or StringIO buffer) and return either the next URL in the pagination
    sequence or None if no more pages exist.
    """
    source = urllib.parse.urlparse(url)
    if source.port == 443:
        conn = http.client.HTTPSConnection(source.hostname, source.port)
    else:
        conn = http.client.HTTPConnection(source.hostname, source.port)
    conn.request("GET", url, headers=headers)
    response = conn.getresponse()
    body = response.read()
    body_json = json.loads(body)
    if response.status != 200:
        print("HTTP error", response.status, body_json)
    elif body_json:
        last_record = {}
        for record in body_json:
            row = transform(record)
            if row:
                out_file.write(row + "\n")
            last_record = record
        print(
            "< {} {} ending on {} {} ({} total)".format(
                len(body_json),
                label,
                last_record["id"],
                last_record["created_at"],
                response.getheader("X-Pagination-Total"),
            )
        )
        return parse_next_value(response.getheader("Link"))
    return None


def get_all_data(url, file_path, label, headers=None, transform=None):
    """
    Repeatedly calls the download function with the next URL in the pagination
    sequence until all data has been read into the StringIO buffer, then writes
    it to disk all at once.
    """
    data = StringIO()
    next_download = lambda u: download(
        u, data, label, headers=headers, transform=transform
    )
    next_page = next_download(url)
    while next_page:
        time.sleep(1)
        next_page = next_download(next_page)
    with open(file_path, "w") as out_file:
        out_file.write(data.getvalue())
    data.close()


if __name__ == "__main__":
    URL_TEMPLATE = "https://io.adafruit.com/api/v2/%s/feeds/%s/data"
    PARAMS = {}
    if os.getenv("START"):
        PARAMS["start_time"] = os.getenv("START")
    elif START_TIME:
        PARAMS["start_time"] = START_TIME
    if os.getenv("END"):
        PARAMS["end_time"] = os.getenv("END")
    elif END_TIME:
        PARAMS["end_time"] = END_TIME

    if not (USERNAME and AIO_KEY):
        print(
            "ERROR: Add your USERNAME, AIO_KEY, and FEEDS values before running this script."
        )
        exit(1)

    HEADERS = {"X-AIO-Key": AIO_KEY}

    for data_label, feed_key, filename in FEEDS:
        filepath = os.path.join(DESTINATION, filename)
        data_url = URL_TEMPLATE % (USERNAME, feed_key)

        if PARAMS:
            data_url += "?" + urllib.parse.urlencode(PARAMS)

        print("---------------------------------------------------------")
        print(
            time.time(), "getting", data_url, "into", filepath, "with HEADERS", HEADERS
        )
        print("---------------------------------------------------------")
        get_all_data(
            data_url, filepath, data_label, headers=HEADERS, transform=TRANSFORM
        )

