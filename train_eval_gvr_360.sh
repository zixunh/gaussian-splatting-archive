set -e
SKIP_TRAIN=false

# Put the sequences under datasets/Ricoh360 this folder. For scannetpp, please put the sequences under datasets/scannetpp
SCENE_ID=bricks #alameda
OUTPUT_SCENE_ID=bricks #alameda_4e-3_2e-6_hsv_grad
DATASET_DIR=/home/choyingw/Documents/ODGS/datasets/Ricoh360/$SCENE_ID/
OUTPUT_DIR=./output/Ricoh360/$OUTPUT_SCENE_ID

STEP_RAW=3e-3
RESIZE_RATIO=1.0
STEP=$(awk 'BEGIN{printf "%.0e", '$STEP_RAW'*'$RESIZE_RATIO'}' | sed 's/e-0*/e-/;s/e+0*/e+/')
FOVMOD_TRAIN=1.0
FOVMOD_EVAL=2.0

STEP_RAW_EVAL=1e-3
RESIZE_RATIO_EVAL=1.0
STEP_EVAL=$(awk 'BEGIN{printf "%.0e", '$STEP_RAW_EVAL'*'$RESIZE_RATIO_EVAL'}' | sed 's/e-0*/e-/;s/e+0*/e+/')

FOVMAP_DIR_TRAIN=fovmaps_fov_"$FOVMOD_TRAIN"_step_"$STEP"/
FOVMAP_DIR_EVAL=fovmaps_fov_"$FOVMOD_EVAL"_step_"$STEP_EVAL"/

TRAIN_MASK_FN=fov_"$FOVMOD_TRAIN"_step_"$STEP"_mask.png
TEST_MASK_FN=fov_"$FOVMOD_EVAL"_step_"$STEP_EVAL"_mask.png

ITERS_NUM=30000
STEP_GT_CROSS_CAM=2e-3

#train
if $SKIP_TRAIN; then
  echo "Load ckpt $ITERS_NUM from output/Ricoh360/$SCENE_ID"
else
  echo "Train $SCENE_ID"
  #python prepare_fov_360.py --path $DATASET_DIR --dst $FOVMAP_DIR_TRAIN --step $STEP_RAW --fov_mod $FOVMOD_TRAIN --mask_dst $TRAIN_MASK_FN --resize_ratio $RESIZE_RATIO
  python train.py -s $DATASET_DIR -m output/Ricoh360/$SCENE_ID \
      --iterations $ITERS_NUM \
      --checkpoint_iterations 15000 30000 \
      --save_iterations 15000 30000 \
      --test_iterations 200 300 500 700 1000 2000 3000 4000 6000 7000 8000 9000 10000 12000 15000 17000 20000 22000 25000 27000 30000 \
      --resolution 1 \
      --eval \
      --sample_step $STEP --fov_mod $FOVMOD_TRAIN \
      --mask_path $DATASET_DIR$FOVMAP_DIR_TRAIN$TRAIN_MASK_FN
    #   --exposure_lr_init 0.001 \
    #   --exposure_lr_final 0.0001 \
    #   --exposure_lr_delay_steps 5000 \
    #   --exposure_lr_delay_mult 0.001 \
    #   --train_test_exp
      #--sibr_mask_refcam "$DATASET_DIR"colmap/cameras_fish.txt 
      # Try to block this flag if you don't want to show mask in the online sibr viewer; Note that we support to render the scene under the mask, while these parts don't affect teh final psnr since they are out of the dataset fov.
fi
