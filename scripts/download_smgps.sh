#!/bin/bash
set -euo pipefail

OUTDIR="files"
HOST="https://archive-gw-1.kat.ac.za/public"
LIST_URL="$HOST/?prefix=repository/10.48479/3wfd-e270/data/MFS_cubes"

mkdir -p "$OUTDIR"
cd "$OUTDIR"

# fetch file list
wget -q -O listing.xml "$LIST_URL"

# extract urls from file
grep -oE '<Key>[^<]+</Key>' listing.xml \
  | sed -E 's#</?Key>##g' \
  | grep -E '\.fits(\.gz)?$' \
  | sed -E "s#^#${HOST}/#g" \
  | sort -u > urls.txt

# download files
wget --continue --tries=3 --timeout=60 --wait=0.5 -i urls.txt
