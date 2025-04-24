set -e
SKIP_TRAIN=true

SCENE_ID=1d003b07bd
DATA_ROOT=/media/scannetpp/demo/
DATASET_DIR=$DATA_ROOT$SCENE_ID/dslr/
OUTPUT_DIR=./output_gvr_undistort/scannetpp/$SCENE_ID

STEP_TRAIN=0.002

FOVMOD_TRAIN=0.85

FOVMAP_DIR_TRAIN=undistorted_tanmap_fov_"$FOVMOD_TRAIN"_step_"$STEP_TRAIN"/


TRAIN_MASK_FN=fov_"$FOVMOD_TRAIN"_step_"$STEP_TRAIN"_mask.png

python prepare_undistorted_pinhole.py --path $DATASET_DIR --dst $FOVMAP_DIR_TRAIN --step $STEP_TRAIN --fov_mod $FOVMOD_TRAIN --mask_dst $TRAIN_MASK_FN