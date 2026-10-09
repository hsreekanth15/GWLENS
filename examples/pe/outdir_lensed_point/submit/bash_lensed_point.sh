#!/usr/bin/env bash

# lensed_point_data0_1126259462-4_generation
# PARENTS 
# CHILDREN lensed_point_data0_1126259462-4_analysis_H1L1
if [[ "lensed_point_data0_1126259462-4_generation" == *"$1"* ]]; then
    echo "Running: /Users/sreekanthharikumar/miniconda3/envs/gwlens/bin/bilby_pipe_generation outdir_lensed_point/lensed_point_config_complete.ini --label lensed_point_data0_1126259462-4_generation --idx 0 --trigger-time 1126259462.4 --outdir outdir_lensed_point"
    /Users/sreekanthharikumar/miniconda3/envs/gwlens/bin/bilby_pipe_generation outdir_lensed_point/lensed_point_config_complete.ini --label lensed_point_data0_1126259462-4_generation --idx 0 --trigger-time 1126259462.4 --outdir outdir_lensed_point
fi

# lensed_point_data0_1126259462-4_analysis_H1L1
# PARENTS lensed_point_data0_1126259462-4_generation
# CHILDREN lensed_point_data0_1126259462-4_analysis_H1L1_final_result
if [[ "lensed_point_data0_1126259462-4_analysis_H1L1" == *"$1"* ]]; then
    echo "Running: /Users/sreekanthharikumar/miniconda3/envs/gwlens/bin/bilby_pipe_analysis outdir_lensed_point/lensed_point_config_complete.ini --outdir outdir_lensed_point --detectors H1 --detectors L1 --label lensed_point_data0_1126259462-4_analysis_H1L1 --data-dump-file outdir_lensed_point/data/lensed_point_data0_1126259462-4_generation_data_dump.pickle --sampler dynesty"
    /Users/sreekanthharikumar/miniconda3/envs/gwlens/bin/bilby_pipe_analysis outdir_lensed_point/lensed_point_config_complete.ini --outdir outdir_lensed_point --detectors H1 --detectors L1 --label lensed_point_data0_1126259462-4_analysis_H1L1 --data-dump-file outdir_lensed_point/data/lensed_point_data0_1126259462-4_generation_data_dump.pickle --sampler dynesty
fi

# lensed_point_data0_1126259462-4_analysis_H1L1_final_result
# PARENTS lensed_point_data0_1126259462-4_analysis_H1L1
# CHILDREN 
if [[ "lensed_point_data0_1126259462-4_analysis_H1L1_final_result" == *"$1"* ]]; then
    echo "Running: /Users/sreekanthharikumar/miniconda3/envs/gwlens/bin/bilby_result --result outdir_lensed_point/result/lensed_point_data0_1126259462-4_analysis_H1L1_result.hdf5 --outdir outdir_lensed_point/final_result --extension hdf5 --max-samples 20000 --lightweight --save"
    /Users/sreekanthharikumar/miniconda3/envs/gwlens/bin/bilby_result --result outdir_lensed_point/result/lensed_point_data0_1126259462-4_analysis_H1L1_result.hdf5 --outdir outdir_lensed_point/final_result --extension hdf5 --max-samples 20000 --lightweight --save
fi

