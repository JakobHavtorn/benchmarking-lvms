#!/usr/bin/env bash

# For a directory, recursively find .flac files and compute 
# - the orignal file size
# - the .flac file size
# - the fraction of the flac file size to the original file (0 < fraction < 1)
# 
# To execute
# > bash evaluate_flac_compression.sh /directory/with/flac > compression_rates.csv
#
# To compute the average fraction, pipe the output to a file and use e.g. awk:
# > cat compression_rates.csv | grep "flac" |  awk -F',' '{sum+=$4; ++n} END { print "Avg: "sum"/"n"="sum/n }'
#
# To compress a directory of .wav files to .flac, run
# > find /some/directory -name '*.wav' -exec flac --best {} \;


echo "file name, original file size, flac file size, flac fraction"

find $1 -name \*.flac \
     | \
     (
      c=1
      while read filename
       do
        # get meta-data from flac file
        meta_data=$(metaflac --show-bps --show-channels --show-total-samples "$filename")

        # get actual file-size
        file_size=$(stat -c '%s' "$filename")

        # compute the raw size (channels*bits_per_samples*total_samples)/8
        # (+7 is for rounding, should it happens)
        #
        original_size=$(echo \(${meta_data[@]}+7\)/8 | tr ' ' '*' | bc)

        # show data and compression ratio (with leading zero)
        echo $filename, $original_size, $file_size, 0$(echo scale=5\;$file_size/$original_size | bc)

        ((c++))
       done
     )
