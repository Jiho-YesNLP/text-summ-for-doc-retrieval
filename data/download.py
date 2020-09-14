"""
download.py

Downloads PubMed annual baseline up to specified number using wget.
"""

import os
from os.path import join as pjoin
import sys
import hashlib
import wget


def check_md5(target_file, md5_file):
    with open(target_file, 'rb') as f:
        md5_ret = hashlib.md5(f.read()).hexdigest()
    true_md5 = open(md5_file).read().split()[-1]
    return md5_ret == true_md5

PUBMED_DIR = 'data/pubmed/'
N_FILES = int(sys.argv[1]) if len(sys.argv) >= 1 else 1015
if not os.path.exists(PUBMED_DIR):
    print(f'mkdir {PUBMED_DIR}')
    os.makedirs(PUBMED_DIR, exist_ok=True)

for num in range(N_FILES):
    filename = f'pubmed20n{num+1:04}.xml.gz'
    filename_md5 = f'pubmed20n{num+1:04}.xml.gz.md5'
    url_base = 'ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/'
    if not os.path.exists(pjoin(PUBMED_DIR, filename_md5)):
        wget.download(url_base + filename_md5, out=PUBMED_DIR)
    if os.path.exists(pjoin(PUBMED_DIR, filename)):
        # check md5
        if check_md5(pjoin(PUBMED_DIR, filename),
                     pjoin(PUBMED_DIR, filename_md5)):
            continue
        print(f'md5 check failed. Removing {filename}...')
        os.remove(pjoin(PUBMED_DIR, filename))

    wget.download(url_base + filename, out=PUBMED_DIR)
    if check_md5(pjoin(PUBMED_DIR, filename),
                 pjoin(PUBMED_DIR, filename_md5)):
    else:
        print(f'md5 check failed. {filename}')
