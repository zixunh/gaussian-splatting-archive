set -e
SKIP_TRAIN=false

# Put the sequences under datasets/zipnerf this folder. For scannetpp, please put the sequences under datasets/scannetpp
SCENE_ID=berlin #alameda
OUTPUT_SCENE_ID=berlin_3e-3_5e-7_hsv_grad #alameda_4e-3_2e-6_hsv_grad
DATASET_DIR=/home/choyingw/Documents/0221_clone/gaussian-splatting/datasets/zipnerf/$SCENE_ID/
DATASET_DIR_PINHOLE=/home/choyingw/Documents/0221_clone/gaussian-splatting/datasets/zipnerf/official_undistorted/$SCENE_ID/
OUTPUT_DIR=./output/zipnerf/$OUTPUT_SCENE_ID

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

#python prepare_fov_zipnerf.py --path $DATASET_DIR --dst $FOVMAP_DIR_TRAIN --step $STEP_RAW --fov_mod $FOVMOD_TRAIN --mask_dst $TRAIN_MASK_FN --resize_ratio $RESIZE_RATIO

#train
# if $SKIP_TRAIN; then
#   echo "Load ckpt $ITERS_NUM from output/zipnerf/$SCENE_ID"
# else
#   echo "Train $SCENE_ID"
#   python prepare_fov_zipnerf.py --path $DATASET_DIR --dst $FOVMAP_DIR_TRAIN --step $STEP_RAW --fov_mod $FOVMOD_TRAIN --mask_dst $TRAIN_MASK_FN --resize_ratio $RESIZE_RATIO
#   python train.py -s $DATASET_DIR -m output/zipnerf/$SCENE_ID \
#       --iterations $ITERS_NUM \
#       --checkpoint_iterations 15000 30000 \
#       --save_iterations 15000 30000 \
#       --test_iterations 200 300 500 700 1000 2000 3000 4000 6000 7000 8000 9000 10000 12000 15000 17000 20000 22000 25000 27000 30000 \
#       --resolution 1 \
#       --eval \
#       --sample_step $STEP --fov_mod $FOVMOD_TRAIN \
#       --mask_path $DATASET_DIR$FOVMAP_DIR_TRAIN$TRAIN_MASK_FN \
#       --exposure_lr_init 0.001 \
#       --exposure_lr_final 0.0001 \
#       --exposure_lr_delay_steps 5000 \
#       --exposure_lr_delay_mult 0.001 \
#       --train_test_exp
#       #--sibr_mask_refcam "$DATASET_DIR"colmap/cameras_fish.txt 
#       # Try to block this flag if you don't want to show mask in the online sibr viewer; Note that we support to render the scene under the mask, while these parts don't affect teh final psnr since they are out of the dataset fov.
# fi

# eval
#python prepare_fov_zipnerf.py --path $DATASET_DIR --dst $FOVMAP_DIR_EVAL --step $STEP_RAW_EVAL --fov_mod $FOVMOD_EVAL --mask_dst $TEST_MASK_FN --resize_ratio $RESIZE_RATIO_EVAL
#render
# python render.py \
#     -m output/zipnerf/$OUTPUT_SCENE_ID \
#     -s $DATASET_DIR \
#     --iteration $ITERS_NUM \
#     --camera_model FISHEYE \
#     --skip_train \
#     --mask_path $DATASET_DIR$FOVMAP_DIR_EVAL$TEST_MASK_FN \
#     --sample_step $STEP_RAW_EVAL --fov_mod $FOVMOD_EVAL \
#     --orig_data_path $DATASET_DIR/images_8

# # # wrap back to origianal space
# python extract_kb_zipnerf.py --path $DATASET_DIR \
#                      --src output/zipnerf/$SCENE_ID/test/ours_$ITERS_NUM/gt \
#                      --dst output/zipnerf/$SCENE_ID/test/ours_$ITERS_NUM/gt_remap \
#                      --step $STEP_EVAL --fov_mod $FOVMOD_EVAL \
#                      -r 8 -resize_factor 4
# python extract_kb_zipnerf.py --path $DATASET_DIR \
#                      --src output/zipnerf/$OUTPUT_SCENE_ID/test/ours_$ITERS_NUM/renders \
#                      --dst output/zipnerf/$OUTPUT_SCENE_ID/test/ours_$ITERS_NUM/renders_remap \
#                      --step $STEP_EVAL --fov_mod $FOVMOD_EVAL \
#                      -r 8 -resize_factor 4

# evaluation
# python metrics.py \
#     -m output/zipnerf/$OUTPUT_SCENE_ID --use_remap



# Cross camera
# Perspective to FOV doesn't need mask actually. 
# python prepare_pers2fov.py --path $DATASET_DIR_PINHOLE \
#     --dst $OUTPUT_DIR/test/ours_$ITERS_NUM/gt_cross_camera --step $STEP_GT_CROSS_CAM \
#     --mask_dst ../$TRAIN_MASK_FN 

python render.py \
    -m output/zipnerf/$OUTPUT_SCENE_ID \
    -s $DATASET_DIR \
    --iteration $ITERS_NUM \
    --camera_model FISHEYE \
    --skip_train \
    --mask_path $DATASET_DIR$FOVMAP_DIR_EVAL$TEST_MASK_FN \
    --sample_step $STEP_RAW_EVAL --fov_mod $FOVMOD_EVAL \
    --get_cross_cam $DATASET_DIR_PINHOLE 

# wrap back to origianal space
python extract_pers_from_fov.py --path $DATASET_DIR_PINHOLE \
                     --src output/zipnerf/$OUTPUT_SCENE_ID/test/ours_$ITERS_NUM/gt_cross_camera \
                     --dst output/zipnerf/$OUTPUT_SCENE_ID/test/ours_$ITERS_NUM/gt_cross_camera_remap \
                     --step $STEP_GT_CROSS_CAM \
                     --skip 8 \
                     -r 4
python extract_pers_from_fov.py --path $DATASET_DIR_PINHOLE \
                     --src output/zipnerf/$OUTPUT_SCENE_ID/test/ours_$ITERS_NUM/renders_cross_camera\
                     --dst output/zipnerf/$OUTPUT_SCENE_ID/test/ours_$ITERS_NUM/renders_cross_camera_remap\
                     --step $STEP_EVAL \
                     -r 4

# evaluation
python metrics_cross_cam.py \
    --output output/zipnerf/$OUTPUT_SCENE_ID/test/ours_$ITERS_NUM/renders_cross_camera_remap \
    --gt output/zipnerf/$OUTPUT_SCENE_ID/test/ours_$ITERS_NUM/gt_cross_camera_remap