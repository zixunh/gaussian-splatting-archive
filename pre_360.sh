set -e

SCENE_ID=bricks 
DATASET_DIR=./$SCENE_ID/

STEP_RAW=3e-3
RESIZE_RATIO=1.0
STEP=$(awk 'BEGIN{printf "%.0e", '$STEP_RAW'*'$RESIZE_RATIO'}' | sed 's/e-0*/e-/;s/e+0*/e+/')
FOVMOD_TRAIN=1.0

FOVMAP_DIR_TRAIN=fovmaps_fov_"$FOVMOD_TRAIN"_step_"$STEP"/

TRAIN_MASK_FN=fov_"$FOVMOD_TRAIN"_step_"$STEP"_mask.png

python prepare_fov_360.py --path $DATASET_DIR --dst $FOVMAP_DIR_TRAIN --step $STEP_RAW --fov_mod $FOVMOD_TRAIN --mask_dst $TRAIN_MASK_FN --resize_ratio $RESIZE_RATIO
