set -e
SKIP_TRAIN=false

# Put the sequences under datasets/zipnerf this folder. For scannetpp, please put the sequences under datasets/scannetpp
SCENE_ID=fisheye/alameda
DATASET_DIR=/media/zipnerf/$SCENE_ID/
OUTPUT_DIR=./output/zipnerf/$SCENE_ID

STEP_RAW=1e-3
RESIZE_RATIO=3.0
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

# train
if $SKIP_TRAIN; then
  echo "Load ckpt $ITERS_NUM from output/zipnerf/$SCENE_ID"
else
  echo "Train $SCENE_ID"
  python prepare_fov_zipnerf.py --path $DATASET_DIR --dst $FOVMAP_DIR_TRAIN --step $STEP_RAW --fov_mod $FOVMOD_TRAIN --mask_dst $TRAIN_MASK_FN --resize_ratio $RESIZE_RATIO
  python train.py -s $DATASET_DIR -m output/zipnerf/$SCENE_ID \
      --iterations $ITERS_NUM \
      --checkpoint_iterations 3000 15000 27000 30000 \
      --save_iterations 3000 15000 27000 30000 \
      --test_iterations 200 300 500 700 1000 2000 3000 4000 6000 7000 8000 9000 10000 12000 15000 17000 20000 22000 25000 27000 30000 \
      --resolution 1 \
      --eval \
      --sample_step $STEP --fov_mod $FOVMOD_TRAIN \
      --mask_path $DATASET_DIR$FOVMAP_DIR_TRAIN$TRAIN_MASK_FN \
      --exposure_lr_init 0.001 \
      --exposure_lr_final 0.0001 \
      --exposure_lr_delay_steps 5000 \
      --exposure_lr_delay_mult 0.001 \
      --train_test_exp
      #--sibr_mask_refcam "$DATASET_DIR"colmap/cameras_fish.txt 
      # Try to block this flag if you don't want to show mask in the online sibr viewer; Note that we support to render the scene under the mask, while these parts don't affect teh final psnr since they are out of the dataset fov.
fi

# # eval
python prepare_fov_zipnerf.py --path $DATASET_DIR --dst $FOVMAP_DIR_EVAL --step $STEP_RAW_EVAL --fov_mod $FOVMOD_EVAL --mask_dst $TEST_MASK_FN --resize_ratio $RESIZE_RATIO_EVAL
# render
python render.py \
    -m output/zipnerf/$SCENE_ID \
    -s $DATASET_DIR \
    --iteration $ITERS_NUM \
    --camera_model FISHEYE \
    --skip_train \
    --mask_path $DATASET_DIR$FOVMAP_DIR_EVAL$TEST_MASK_FN \
    --sample_step $STEP_RAW_EVAL --fov_mod $FOVMOD_EVAL \
    --orig_data_path $DATASET_DIR/images_8

# wrap back to origianal space
python extract_kb_zipnerf.py --path $DATASET_DIR \
                     --src output/zipnerf/$SCENE_ID/test/ours_$ITERS_NUM/gt \
                     --dst output/zipnerf/$SCENE_ID/test/ours_$ITERS_NUM/gt_remap \
                     --step $STEP_EVAL --fov_mod $FOVMOD_EVAL \
                     -r 8 -resize_factor 4
python extract_kb_zipnerf.py --path $DATASET_DIR \
                     --src output/zipnerf/$SCENE_ID/test/ours_$ITERS_NUM/renders \
                     --dst output/zipnerf/$SCENE_ID/test/ours_$ITERS_NUM/renders_remap \
                     --step $STEP_EVAL --fov_mod $FOVMOD_EVAL \
                     -r 8 -resize_factor 4

# evaluation
python metrics.py \
    -m output/zipnerf/$SCENE_ID --use_remap