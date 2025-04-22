set -e
SKIP_TRAIN=false

# Put the sequences under datasets/zipnerf this folder. For scannetpp, please put the sequences under datasets/scannetpp
SCENE_ID=alameda #alameda
DATASET_DIR=/home/choyingw/Documents/0221_clone/gaussian-splatting/datasets/zipnerf/$SCENE_ID/
DATASET_DIR_PINHOLE=/home/choyingw/Documents/0221_clone/gaussian-splatting/datasets/zipnerf/official_undistorted/$SCENE_ID/

STEP_RAW=1e-3
RESIZE_RATIO=4.0
STEP=$(awk 'BEGIN{printf "%.0e", '$STEP_RAW'*'$RESIZE_RATIO'}' | sed 's/e-0*/e-/;s/e+0*/e+/')
FOVMOD_TRAIN=1.0
FOVMOD_EVAL=2.0

STEP_RAW_EVAL=1e-3
RESIZE_RATIO_EVAL=1.0
STEP_EVAL=$(awk 'BEGIN{printf "%.0e", '$STEP_RAW_EVAL'*'$RESIZE_RATIO_EVAL'}' | sed 's/e-0*/e-/;s/e+0*/e+/')

FOVMAP_DIR_TRAIN=undistorted_fovmaps_fov_"$FOVMOD_TRAIN"_step_"$STEP"/
FOVMAP_DIR_EVAL=undistorted_fovmaps_fov_"$FOVMOD_EVAL"_step_"$STEP_EVAL"/

TRAIN_MASK_FN=fov_"$FOVMOD_TRAIN"_step_"$STEP"_mask.png
TEST_MASK_FN=fov_"$FOVMOD_EVAL"_step_"$STEP_EVAL"_mask.png

ITERS_NUM=30000
STEP_GT_CROSS_CAM=2e-3

# eval
python prepare_fov_zipnerf.py --path $DATASET_DIR --dst $FOVMAP_DIR_EVAL --step $STEP_RAW_EVAL --fov_mod $FOVMOD_EVAL --mask_dst $TEST_MASK_FN --resize_ratio $RESIZE_RATIO_EVAL
#render
python render.py \
    -m /home/choyingw/Documents/0404_pinhole/gaussian-splatting-gvr/output/$SCENE_ID \
    -s $DATASET_DIR \
    --iteration $ITERS_NUM \
    --camera_model FISHEYE \
    --skip_train \
    --mask_path $DATASET_DIR$FOVMAP_DIR_EVAL$TEST_MASK_FN \
    --sample_step $STEP_RAW_EVAL --fov_mod $FOVMOD_EVAL \
    --orig_data_path $DATASET_DIR/images_8

# # # wrap back to origianal space
python extract_kb_zipnerf.py --path $DATASET_DIR \
                     --src /home/choyingw/Documents/0404_pinhole/gaussian-splatting-gvr/output/$SCENE_ID/test/ours_$ITERS_NUM/gt_cross_camera \
                     --dst /home/choyingw/Documents/0404_pinhole/gaussian-splatting-gvr/output/$SCENE_ID/test/ours_$ITERS_NUM/gt_cross_camera_remap \
                     --step $STEP_EVAL --fov_mod $FOVMOD_EVAL \
                     -r 8 -resize_factor 4
python extract_kb_zipnerf.py --path $DATASET_DIR \
                     --src /home/choyingw/Documents/0404_pinhole/gaussian-splatting-gvr/output/$SCENE_ID/test/ours_$ITERS_NUM/renders_cross_camera \
                     --dst /home/choyingw/Documents/0404_pinhole/gaussian-splatting-gvr/output/$SCENE_ID/test/ours_$ITERS_NUM/renders_cross_camera_remap \
                     --step $STEP_EVAL --fov_mod $FOVMOD_EVAL \
                     -r 8 -resize_factor 4

# evaluation
python metrics.py \
    -m /home/choyingw/Documents/0404_pinhole/gaussian-splatting-gvr/output/$SCENE_ID --use_remap

