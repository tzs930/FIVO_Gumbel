#!/bin/bash
# A script to download the pianoroll datasets.
# Accepts one argument, the directory to put the files in

if [ -z "$1" ]
  then
    echo "Error, must provide a directory to download the files to."
    exit
fi

echo "Downloading datasets into $1"
curl -s "http://www-etud.iro.umontreal.ca/~boulanni/Piano-midi.de.pickle" > $1/piano-midi.de.pkl
curl -s "http://www-etud.iro.umontreal.ca/~boulanni/Nottingham.pickle" > $1/nottingham.pkl
curl -s "http://www-etud.iro.umontreal.ca/~boulanni/MuseData.pickle" > $1/musedata.pkl
curl -s "http://www-etud.iro.umontreal.ca/~boulanni/JSB%20Chorales.pickle" > $1/jsb.pkl
