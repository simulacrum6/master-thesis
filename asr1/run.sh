#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1
. ./cmd.sh || exit 1

# general configuration
backend=pytorch
stage=-1 # start from -1 if you need to start from data preparation
stop_stage=6
ngpu=4   # number of gpus ("0" uses cpu, otherwise use gpu)
njobs=14 # number of jobs to use for cpu tasks
debugmode=1
N=0       # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0 # verbose option
resume=   # Resume the training from snapshot
#n-iter-processes=2 # enable data prefetching, kills ram though

# feature configuration
do_delta=false
train_config=conf/LAS-tuned.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
n_average=10               # used for transformers. averages over n last models.

# dirs
dsname=libri # name of the dataset
tuning=false # whether the current run is for hyperparameter tuning
aug_suffix="x"
max_aug_factor="8.0x"     # maximum number of replications for augmentation
intensity="medium"        # description of augmentation intensity
tag="final"               # tag for managing experiments.

dldir=downloads/${dsname} # directory for raw data and downloads
datadir=data/${dsname}    # directory for processed data (kaldi style)
fbankdir=fbank/${dsname}  # directory for fbank features
dumpdir=dump/${dsname}    # directory to dump full features
expdir=exp/${dsname}      # directory for experiment results

# LibriSpeech Dataset
train_set="train-clean-100"
dev_set="dev-clean"
test_set="test-clean"
dialect="tsv"

# CommonVoice Dataset
lang="en"
data_url=https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3
cvdir=downloads/commonvoice/${lang}

if [ $tuning ]; then
    recog_set=$dev_set
else
    recog_set="$test_set"
fi

. utils/parse_options.sh || exit 1

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "STAGE -1: Preprocessing"

    data_url_libri=www.openslr.org/resources/12
    libri_parts="dev-clean test-clean dev-other test-other train-clean-100"
    tmpdir="${dldir}/tmp/"
    clipdir="${dldir}/clips/"
    if [ ! -d "${tmpdir}" ] && [ ! -d "${clipdir}" ]; then
        echo "downloading LibriSpeech dataset to ${tmpdir}"
        mkdir -p ${tmpdir}
        for part in ${libri_parts}; do
            local/download_and_untar_libri.sh ${tmpdir} ${data_url_libri} ${part}
        done
    else
        echo "data already downloaded. skipping download."
    fi

    if [ ! -d ${clipdir} ]; then
        echo "converting from LibriSpeech to CommonVoice format."
        python local/librispeech2commonvoice.py ${tmpdir} ${dldir} ${libri_parts}
    else
        echo "data already converted. skipping conversion."
    fi

    # download commonvoice
    mkdir -p ${cvdir}
    local/download_and_untar_commonvoice.sh ${cvdir} ${data_url}/${lang}.tar.gz ${lang}.tar.gz
fi

###
# DATA PREPARATION
###
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data Preparation"

    # extend dataset with unique ids and durations
    echo "extending datasets"
    for ds in $train_set $dev_set $test_set; do
        python local/extend_dataset.py ${dldir} "${ds}.${dialect}"
    done

    # use extended datasets instead of regular ones.
    train_set="${train_set}_ext"
    dev_set="${dev_set}_ext"
    test_set="${test_set}_ext"
    recog_set="${test_set}"

    # generate subsets
    echo "creating subsets"
    python local/create_subsets.py ${dldir} ${train_set}.${dialect}

    # augment data
    aug_set="aug_${train_set}_${max_aug_factor}_${intensity}"
    if [ ! -f ${dldir}/${aug_set}.${dialect} ]; then
        echo "augmenting data"
        python local/augment_dataset.py ${dldir} ${train_set}.${dialect}
    else
        echo "data already augmented. skipping augmentation."
    fi

    # generate data dir
    echo "preparing data dir"
    feature_sets="${train_set} ${dev_set} ${test_set} ${aug_set}"
    for ds in ${feature_sets}; do
        echo "preparing data dir for $ds"
        mkdir -p ${datadir}/${ds}
        local/data_prep.pl ${dldir} ${ds} ${datadir}/${ds}
    done

    # generate data for commonvoice dataset
    echo "generating data for commonvoice"
    mkdir -p data/commonvoice/${lang}/validated
    local/data_prep_orig.pl ${tmpdir} "validated" data/commonvoice/${lang}/validated
    local/split_tr_dt_et.sh data/commonvoice/${lang}/validated data/commonvoice/${lang}/train data/commonvoice/${lang}/dev data/commonvoice/${lang}/test
fi

if [ ${stage} -ge 1 ]; then
    # use extended datasets instead of regular ones.
    train_set="${train_set}_ext"
    dev_set="${dev_set}_ext"
    test_set="${test_set}_ext"
    recog_set="${test_set}"
    aug_set="aug_${train_set}_${max_aug_factor}_${intensity}"
    feature_sets="${train_set} ${dev_set} ${test_set} ${aug_set}"
fi

feat_train_dir=${dumpdir}/${train_set}/delta${do_delta}
mkdir -p ${feat_train_dir}
feat_dev_dir=${dumpdir}/${dev_set}/delta${do_delta}
mkdir -p ${feat_dev_dir}
feat_aug_dir=${dumpdir}/${aug_set}/delta${do_delta}
mkdir -p ${feat_aug_dir}
feat_recog_dir_cv=dump/commonvoice/${lang}/delta${do_delta}
mkdir -p ${feat_recog_dir_cv}
###
# FEATURE GENERATION
###
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "STAGE 1: Feature Generation"
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    echo "generating raw fbank features"
    for x in ${feature_sets}; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj $njobs --write_utt2num_frames true \
        ${datadir}/${x} ${expdir}/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh ${datadir}/${x}
    done

    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 14 --write_utt2num_frames true \
    data/commonvoice/${lang}/test exp/make_fbank/commonvoice/${lang}/test "fbank"
    utils/fix_data_dir.sh data/commonvoice/${lang}/test

    # compute global CMVN
    echo "computing Cepstral Mean Variance Normalization"
    compute-cmvn-stats scp:${datadir}/${train_set}/feats.scp ${datadir}/${train_set}/cmvn.ark
    compute-cmvn-stats scp:${datadir}/${aug_set}/feats.scp ${datadir}/${aug_set}/cmvn.ark

    # creating features
    echo "generating complete features"
    dump.sh --cmd "$train_cmd" --nj $njobs --do_delta ${do_delta} \
    ${datadir}/${train_set}/feats.scp ${datadir}/${train_set}/cmvn.ark ${expdir}/dump_feats/train ${feat_train_dir}
    dump.sh --cmd "$train_cmd" --nj $njobs --do_delta ${do_delta} \
    ${datadir}/${dev_set}/feats.scp ${datadir}/${train_set}/cmvn.ark ${expdir}/dump_feats/dev ${feat_dev_dir}
    dump.sh --cmd "$train_cmd" --nj $njobs --do_delta ${do_delta} \
    ${datadir}/${aug_set}/feats.scp ${datadir}/${aug_set}/cmvn.ark ${expdir}/dump_feats/aug ${feat_aug_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj $njobs --do_delta ${do_delta} \
        ${datadir}/${rtask}/feats.scp ${datadir}/${train_set}/cmvn.ark ${expdir}/dump_feats/recog/${rtask} \
        ${feat_recog_dir}
    done

    # create data for commonvoice
    dump.sh --cmd "$train_cmd" --nj 14 --do_delta ${do_delta} \
    data/commonvoice/${lang}/test/feats.scp \
    ${datadir}/${train_set}/cmvn.ark exp/dump_feats/recog/commonvoice/${lang}/test \
    ${feat_recog_dir_cv}

    # create augmented subsets and augmentation only sets
    echo "creating directories for augmented subsets"
    python local/create_kaldi_experiment_dirs.py ${dldir} ${train_set} ${datadir} ${do_delta}

    # loop over augmented directories
    full_train_dir="${datadir}/${train_set}"
    max_aug_dir="${datadir}/${aug_set}"
    regular_ids="regular.txt"
    augmented_ids="augmented.txt"
    for dir in ${datadir}/${train_set}_*; do
        local/generate_kaldi_files.sh ${dir} ${regular_ids} ${augmented_ids} ${full_train_dir} ${max_aug_dir} \
        ${dumpdir} ${do_delta} ${feat_train_dir} ${feat_aug_dir} &
    done
    wait
fi

###
# DICT AND JSON DATA
###
# bpemode (unigram or bpe)
nbpe=30         # ascii alphabet + ' + special tokens <eos> <blank>
bpemode=unigram # unigram or bpe

lcdir=${datadir}/lang_char
input=${lcdir}/input.txt
dict=${lcdir}/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=${lcdir}/${train_set}_${bpemode}${nbpe}
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "STAGE 2: Dictionary and Json Data Preparation"
    mkdir -p ${lcdir}/
    echo "<unk> 1" >${dict} # <unk> must be 1, 0 will be used for "blank" in CTC

    echo "creating Subword Dictionary using $bpemode"
    cut -f 2- -d" " ${datadir}/${train_set}/text >${input}
    spm_train --input=${input} --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
    spm_encode --model=${bpemodel}.model --output_format=piece <${input} | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >>${dict}
    wc -l ${dict}

    echo "make train json files"
    for feat_dir in ${dumpdir}/${train_set}*; do
        (
            ts=$(basename ${feat_dir})
            feat_dir=${feat_dir}/delta${do_delta}

            data2json.sh --feat ${feat_dir}/feats.scp --bpecode ${bpemodel}.model \
            ${datadir}/${ts} ${dict} >${feat_dir}/data_${bpemode}${nbpe}.json
        ) &
    done
    wait

    echo "make dev and train json files"
    data2json.sh --feat ${feat_dev_dir}/feats.scp --bpecode ${bpemodel}.model \
    ${datadir}/${dev_set} ${dict} >${feat_dev_dir}/data_${bpemode}${nbpe}.json
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp --bpecode ${bpemodel}.model \
        ${datadir}/${rtask} ${dict} >${feat_recog_dir}/data_${bpemode}${nbpe}.json
    done

    echo "make ood files"
    data2json.sh --feat ${feat_recog_dir_cv}/feats.scp --bpecode ${bpemodel}.model \
    data/commonvoice/${lang}/test ${dict} >${feat_recog_dir_cv}/data_${bpemode}${nbpe}.json
    rm -r ${dumpdir}/ood
    mkdir -p ${dumpdir}/ood
    cp -r dump/commonvoice/${lang}/. ${dumpdir}/ood
fi

###
# NETWORK TRAINING
###
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "STAGE 4: Network Training"
    ###
    # Regular Runs
    ###
    for feat_dir in ${dumpdir}/${train_set}*; do
        ts=$(basename ${feat_dir})
        feat_dir=${feat_dir}/delta${do_delta}
        expname=${dsname}_${ts}_$(basename ${train_config%.*})_${nbpe}${bpemode}_${tag}
        ed=${expdir}/${expname}
        mkdir -p ${ed}

        if [ ! -f ${ed}/.done ]; then
            echo "training $ts model"
            ${cuda_cmd} --gpu ${ngpu} ${ed}/train.log \
            asr_train.py \
            --config ${train_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --outdir ${ed}/results \
            --tensorboard-dir tensorboard/${expname} \
            --debugmode ${debugmode} \
            --dict ${dict} \
            --debugdir ${ed} \
            --minibatches ${N} \
            --verbose ${verbose} \
            --resume ${resume} \
            --train-json ${feat_dir}/data_${bpemode}${nbpe}.json \
            --valid-json ${feat_dev_dir}/data_${bpemode}${nbpe}.json

            echo -n >${ed}/.done
        else
            echo "training for $ts already complete. skipping training"
        fi
    done

    ###
    # EXTENDED RUNS
    # Promising model is trained for the same number of optimization steps as 100hr baseline
    ###
    for feat_dir in ${dumpdir}/${train_set}_10hrs ${dumpdir}/${train_set}_10hrs_1.0x_medium; do
        ts=$(basename ${feat_dir})
        feat_dir=${feat_dir}/delta${do_delta}
        expname=${dsname}_${ts}_$(basename ${train_config%.*})_${nbpe}${bpemode}_extended
        ed=${expdir}/${expname}
        mkdir -p ${ed}
        if [[ "${ts}" == ${train_set}_10hrs ]]; then
            epochs=250
            patience=20
        else
            epochs=125
            patience=10
        fi

        if [ ! -f ${ed}/.done ]; then
            echo "training $ts model"
            ${cuda_cmd} --gpu ${ngpu} ${ed}/train.log \
            asr_train.py \
            --config ${train_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --outdir ${ed}/results \
            --tensorboard-dir tensorboard/${expname} \
            --debugmode ${debugmode} \
            --dict ${dict} \
            --debugdir ${ed} \
            --minibatches ${N} \
            --verbose ${verbose} \
            --resume ${resume} \
            --train-json ${feat_dir}/data_${bpemode}${nbpe}.json \
            --valid-json ${feat_dev_dir}/data_${bpemode}${nbpe}.json \
            --patience ${patience} \
            --epochs ${epochs}

            echo -n >${ed}/.done
        else
            echo "training for $ts already complete. skipping training"
        fi
    done
fi

###
# DECODING
###
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "STAGE 5: Decoding"
    ngpu=0 # cpu decoding
    echo "$(date)"

    ###
    # Regular Runs
    ###
    for feat_dir in ${dumpdir}/${train_set}*; do
        ts=$(basename ${feat_dir}) # train set
        if [[ "${ts}" == train-clean-100_ext_0.05hrs* ]]; then
            echo "skipping $ts. faulty model."
            continue
        fi

        feat_dir=${feat_dir}/delta${do_delta}
        expname=${dsname}_${ts}_$(basename ${train_config%.*})_${nbpe}${bpemode}_${tag}
        ed=${expdir}/${expname} # experiment dir
        pids=()                 # initialize pids

        echo "$(date)"
        echo "decoding using $ts model"
        for rtask in test-clean_ext ood dev-clean_ext; do
            decode_dir=decode_${rtask}_$(basename ${decode_config%.*})
            feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
            if [ ! -f ${ed}/${decode_dir}/.done ]; then
                echo "decoding $rtask."
                (
                    # split up tasks
                    splitjson.py --parts ${njobs} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

                    # decode
                    ${decode_cmd} JOB=1:${njobs} ${ed}/${decode_dir}/log/decode.JOB.log \
                    asr_recog.py \
                    --config ${decode_config} \
                    --ngpu ${ngpu} \
                    --backend ${backend} \
                    --batchsize 0 \
                    --recog-json ${feat_recog_dir}/split${njobs}utt/data_${bpemode}${nbpe}.JOB.json \
                    --result-label ${ed}/${decode_dir}/data.JOB.json \
                    --model ${ed}/results/${recog_model}

                    # score model
                    score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${ed}/${decode_dir} ${dict}

                    echo -n >${ed}/${decode_dir}/.done
                ) &
                pids+=($!) # store background pids
            else
                echo "decoding for $rtask already complete. skipping decoding."
            fi
        done
        i=0
        for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
        [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    done

    ###
    # EXTENDED RUNS
    # Promising model is trained for the same number of optimization steps as 100hr baseline
    ###
    for feat_dir in ${dumpdir}/${train_set}_10hrs ${dumpdir}/${train_set}_10hrs_1.0x_medium; do
        feat_dir=${feat_dir}/delta${do_delta}
        expname=${dsname}_${ts}_$(basename ${train_config%.*})_${nbpe}${bpemode}_extended
        ed=${expdir}/${expname} # experiment dir
        pids=()                 # initialize pids

        echo "$(date)"
        echo "decoding using $ts model"
        for rtask in test-clean_ext ood dev-clean_ext; do
            decode_dir=decode_${rtask}_$(basename ${decode_config%.*})
            feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
            if [ ! -f ${ed}/${decode_dir}/.done ]; then
                echo "decoding $rtask."
                (
                    # split up tasks
                    splitjson.py --parts ${njobs} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

                    # decode
                    ${decode_cmd} JOB=1:${njobs} ${ed}/${decode_dir}/log/decode.JOB.log \
                    asr_recog.py \
                    --config ${decode_config} \
                    --ngpu ${ngpu} \
                    --backend ${backend} \
                    --batchsize 0 \
                    --recog-json ${feat_recog_dir}/split${njobs}utt/data_${bpemode}${nbpe}.JOB.json \
                    --result-label ${ed}/${decode_dir}/data.JOB.json \
                    --model ${ed}/results/${recog_model}

                    # score model
                    score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${ed}/${decode_dir} ${dict}

                    echo -n >${ed}/${decode_dir}/.done
                ) &
                pids+=($!) # store background pids
            else
                echo "decoding for $rtask already complete. skipping decoding."
            fi
        done
        i=0
        for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
        [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    done
    echo "Finished"
fi

###
# Result Aggregation
###
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    resultdir=results
    rawdir=${resultdir}/raw
    decodedir=${rawdir}/decode
    mkdir -p ${resultdir}
    mkdir -p ${rawdir}
    mkdir -p ${decodedir}

    for d in ${expdir}/*/decode*; do
        if [ -f ${d}/.done ]; then
            echo "converting results from ${d}"
            python local/results2tsv.py ${d} ${decodedir}
        else
            echo "${d} not decoded yet. skipping directory."
        fi
    done

    echo "merging results from ${decodedir} in ${rawdir}"
    python local/merge_results.py ${decodedir} ${rawdir} results wer_stats cer_stats
fi
