set -e
SKIP_TRAIN=true

SCENE_ID=4ef75031e3
DATA_ROOT=/media/scannetpp/demo/
DATASET_DIR=$DATA_ROOT$SCENE_ID/dslr/

STEP_TRAIN=0.002
STEP_EVAL=0.0015

FOVMOD_TRAIN=1.0
FOVMOD_EVAL=1.0

FOVMAP_DIR_TRAIN=undistorted_images/ #undistorted_fovmap_fov_"$FOVMOD_TRAIN"_step_"$STEP_TRAIN"/
FOVMAP_DIR_EVAL=undistorted_images/ #undistorted_fovmap_fov_"$FOVMOD_EVAL"_step_"$STEP_EVAL"/

ITERS_NUM=30000

# if [ "$SCENE_ID" = "1d003b07bd" ]; then
#     FOVMOD_TRAIN=0.85
#     FOVMOD_EVAL=0.85
# else
#     FOVMOD_TRAIN=1.0
#     FOVMOD_EVAL=1.0
# fi

if [ "$FOVMOD_TRAIN" = "1.0" ]; then
    # OUTPUT_DIR="./output_3dgs_undistort/scannetpp/$SCENE_ID"
    OUTPUT_DIR="./output_gvr_undistort/scannetpp/$SCENE_ID"
else
    # OUTPUT_DIR="./output_3dgs_undistort/scannetpp_fov$FOVMOD_TRAIN/$SCENE_ID"
    OUTPUT_DIR="./output_gvr_undistort/scannetpp_fov$FOVMOD_TRAIN/$SCENE_ID"
fi

TRAIN_MASK_FN=fov_"$FOVMOD_TRAIN"_step_"$STEP_TRAIN"_mask.png
TEST_MASK_FN=fov_"$FOVMOD_EVAL"_step_"$STEP_EVAL"_mask.png

# train
if $SKIP_TRAIN; then
  echo "Load ckpt $ITERS_NUM from output/scannetpp/$SCENE_ID"
else
  echo "Train $SCENE_ID"

  python prepare_undistorted_pinhole.py --path $DATASET_DIR --dst $FOVMAP_DIR_TRAIN --step $STEP_TRAIN --fov_mod $FOVMOD_TRAIN --mask_dst $TRAIN_MASK_FN
  python train.py -s $DATASET_DIR -m $OUTPUT_DIR \
      --iterations $ITERS_NUM \
      --checkpoint_iterations 3000 7000 15000 30000 \
      --save_iterations 3000 7000 15000 30000 \
      --test_iterations 200 300 500 700 1000 2000 3000 4000 7000 8000 9000 10000 12000 15000 17000 20000 22000 25000 27000 30000 \
      --resolution 1 \
      --eval \
      --sample_step $STEP_TRAIN --fov_mod $FOVMOD_TRAIN \
      --mask_path $DATASET_DIR$FOVMAP_DIR_TRAIN$TRAIN_MASK_FN \
      # --sibr_mask_refcam "$DATASET_DIR"colmap/cameras_fish.txt 
      # Try to block the flag 'sibr_mask_refcam' if you don't want to show mask in the online sibr viewer;
      # Note that we support to render the scene under the mask,
      # while these parts don't affect the final psnr since they are out of the dataset FoV.
fi

# eval
python prepare_undistorted_pinhole.py --path $DATASET_DIR --dst $FOVMAP_DIR_EVAL --step $STEP_EVAL --fov_mod $FOVMOD_EVAL --mask_dst $TEST_MASK_FN
# render
python render.py \
    -m $OUTPUT_DIR \
    -s $DATASET_DIR \
    --iteration $ITERS_NUM \
    --camera_model PINHOLE \
    --skip_train \
    --mask_path $DATASET_DIR$FOVMAP_DIR_EVAL$TEST_MASK_FN \
    --sample_step $STEP_EVAL --fov_mod $FOVMOD_EVAL

# wrap back to origianal space
echo "Ground truth (kb) remapping from FoVMap"
python extract_kb_pinhole.py --path $DATASET_DIR \
                    --src $OUTPUT_DIR/test/ours_$ITERS_NUM/gt \
                    --dst $OUTPUT_DIR/test/ours_$ITERS_NUM/gt_remap \
                    --step $STEP_EVAL --fov_mod $FOVMOD_EVAL --gridmap_restrict

python extract_kb_pinhole.py --path $DATASET_DIR \
                     --src $OUTPUT_DIR/test/ours_$ITERS_NUM/renders \
                     --dst $OUTPUT_DIR/test/ours_$ITERS_NUM/renders_remap \
                     --step $STEP_EVAL --fov_mod $FOVMOD_EVAL --gridmap_restrict

# evaluation
python metrics.py \
    -m $OUTPUT_DIR --use_remap \
    --iters $ITERS_NUM \
    # --custom_gt /home/scannetpp_ever_gt/dslr/$SCENE_ID/test/ours_$ITERS_NUM/gt \
