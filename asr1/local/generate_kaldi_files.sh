#!/bin/bash

dir=$1
regular_ids=$2
augmented_ids=$3
full_train_dir=$4
max_aug_dir=$5
dumpdir=$6
do_delta=$7
feat_train_dir=$8
feat_aug_dir=$9

# skip if nothing to expand
if [ ! -f ${dir}/${regular_ids} ] && [ ! -f ${dir}/${augmented_ids} ]; then
    echo "$dir already complete. skipping $dir."
else
    # replace leading regex marker...
    sed -i 's/\^//' ${dir}/${regular_ids}
    sed -i 's/\^//' ${dir}/${augmented_ids}

    # copy lines from id files of train set and max aug set if they are in regular_ids or augmented_ids
    echo "generating id files for $dir"
    for f in text feats.scp utt2dur utt2num_frames wav.scp utt2gender utt2spk; do
        echo $f
        utils/filter_scp.pl ${dir}/${regular_ids} ${full_train_dir}/${f} >${dir}/${f}
        utils/filter_scp.pl ${dir}/${augmented_ids} ${max_aug_dir}/${f} >>${dir}/${f}
    done
    # copy frame shift and configs
    echo "copying frame_shift and conf"
    for f in frame_shift conf; do
        cp -r ${full_train_dir}/${f} ${dir}/${f}
    done

    echo "generating spk2utt"
    utils/utt2spk_to_spk2utt.pl ${dir}/utt2spk >${dir}/spk2utt
    echo "checking dir integrity"
    utils/fix_data_dir.sh ${dir}

    # create dump dir
    echo "creating dump dir"
    feat_dir=${dumpdir}/$(basename $dir)/delta${do_delta}
    mkdir -p $feat_dir
    cp "${feat_train_dir}/filetype" "${feat_dir}/"
    for f in feats.scp utt2num_frames; do
        echo "filtering $f"
        utils/filter_scp.pl ${dir}/${regular_ids} ${feat_train_dir}/${f} >${feat_dir}/${f}
        utils/filter_scp.pl ${dir}/${augmented_ids} ${feat_aug_dir}/${f} >>${feat_dir}/${f}
        sort -o ${feat_dir}/${f} ${feat_dir}/${f}
    done

    # clean up
    echo "cleaning up"
    rm ${dir}/${regular_ids}
    rm ${dir}/${augmented_ids}
fi
