"""
Automatically download PubMed abstracts.
"""

import ftplib
import re
import os
import gzip
import shutil

# Global constants
PUBMED_FTP = "ftp.ncbi.nlm.nih.gov"
PUBMED_ROUTE = "/pubmed/baseline/"
XML_GZ_REGEXP = ".xml.gz$"
NUMBER_OF_FILES_TO_SCRAPE = 1  # approx 14,000 abstracts per file
TARGET = 'AbstractText'

ftp = ftplib.FTP(PUBMED_FTP)
ftp.login("anonymous", "")
ftp.cwd(PUBMED_ROUTE)
files = []

# Get list of files
try:
    files = ftp.nlst()
except ftplib.error_perm as empty:
    if str(empty) == "550 No files found":
        print("No files in this directory")
    else:
        raise

# keep only XML_GZ
files = list(filter(lambda f: re.search(XML_GZ_REGEXP, f), files))

# create the data folder if not exists
if not os.path.exists("../data"):
    os.makedirs("../data")

abstracts_file = open('../data/abstracts.txt', 'w+')

# limit files to scrape
if (NUMBER_OF_FILES_TO_SCRAPE < len(files)):
    files = files[:NUMBER_OF_FILES_TO_SCRAPE]
print(f'files to scrape: {files}')

number_of_abstracts = 0

for f in files:
    # create the temporary files
    local_f_xml_gz = os.path.join("../data", f)
    local_f_xml = os.path.join("../data", f[:-3])

    # download the xml_gz file
    lf = open(local_f_xml_gz, "wb")
    ftp.retrbinary("RETR " + f, lf.write, 8 * 1024)
    lf.close()

    # unzip it into the xml file
    with gzip.open(local_f_xml_gz, 'rb') as f_in:
        with open(local_f_xml, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    # delete the xml_gz file
    os.remove(local_f_xml_gz)

    # extract abstracts
    with open(local_f_xml, 'r') as f:
        for line in f:
            abstract_search = re.search('\s+<' + TARGET + '>(.*)</' + TARGET + '>', line)

            if abstract_search:
                abstract = abstract_search.group(1)

                abstracts_file.write(abstract)
                abstracts_file.write("\n")

                number_of_abstracts += 1

    print(f'extracted {number_of_abstracts} abstracts from {local_f_xml}')

    # delete the xml file
    os.remove(local_f_xml)

print(f'downloaded {number_of_abstracts} abstracts in total')
abstracts_file.close()