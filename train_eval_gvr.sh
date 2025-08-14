set -e
SKIP_TRAIN=true

SCENE_ID=1d003b07bd #4ef75031e3 #2a1a3afad9 #1f7cbbdde1 4ef75031e3 1d003b07bd
DATA_ROOT=/media/scannetpp/demo/
DATASET_DIR=$DATA_ROOT$SCENE_ID/dslr/
# OUTPUT_DIR=./output_ut_tight/scannetpp/$SCENE_ID
# OUTPUT_DIR=./output_ewa/scannetpp/$SCENE_ID
# OUTPUT_DIR=./output_fov/scannetpp_fov0.85/$SCENE_ID
# OUTPUT_DIR=./output_fov/scannetpp/$SCENE_ID
# OUTPUT_DIR=./output_fullfov_updated/scannetpp/$SCENE_ID
OUTPUT_DIR=../../omni-3dgs/output_achive/scannetpp/$SCENE_ID
# OUTPUT_DIR=../../scannetpp_fs_gt/dslr/$SCENE_ID

STEP_TRAIN=0.002
STEP_EVAL=0.0015

FOVMOD_TRAIN=1.3 #0.85 #0.85 #1.0 #1.3 #1.0 #1.3
FOVMOD_EVAL=2.0 #0.85 #0.85 #1.0 #2.0 #1.0 #2.0

FOVMAP_DIR_TRAIN=undistorted_fovmaps_fov_"$FOVMOD_TRAIN"_step_"$STEP_TRAIN"/
FOVMAP_DIR_EVAL=undistorted_fovmaps_fov_"$FOVMOD_EVAL"_step_"$STEP_EVAL"/

TRAIN_MASK_FN=fov_"$FOVMOD_TRAIN"_step_"$STEP_TRAIN"_mask.png
TEST_MASK_FN=fov_"$FOVMOD_EVAL"_step_"$STEP_EVAL"_mask.png

ITERS_NUM=30000

# train
if $SKIP_TRAIN; then
  echo "Load ckpt $ITERS_NUM from $OUTPUT_DIR"
else
  echo "Train $SCENE_ID"
  python prepare_fov.py --path $DATASET_DIR --dst $FOVMAP_DIR_TRAIN --step $STEP_TRAIN --fov_mod $FOVMOD_TRAIN --mask_dst $TRAIN_MASK_FN
  python train.py -s $DATASET_DIR -m $OUTPUT_DIR \
      --iterations $ITERS_NUM \
      --checkpoint_iterations 300 3000 7000 15000 30000 \
      --save_iterations 300 3000 7000 15000 30000 \
      --test_iterations 200 300 500 700 1000 2000 3000 4000 7000 8000 9000 10000 12000 15000 17000 20000 22000 25000 27000 30000 \
      --resolution 1 \
      --eval \
      --sample_step $STEP_TRAIN --fov_mod $FOVMOD_TRAIN \
      --mask_path $DATASET_DIR$FOVMAP_DIR_TRAIN$TRAIN_MASK_FN \
      --sibr_mask_refcam "$DATASET_DIR"colmap/cameras_fish.txt 
      # Try to block the flag 'sibr_mask_refcam' if you don't want to show mask in the online sibr viewer;
      # Note that we support to render the scene under the mask,
      # while these parts don't affect the final psnr since they are out of the dataset FoV.
fi

# eval
# python prepare_fov.py --path $DATASET_DIR --dst $FOVMAP_DIR_EVAL --step $STEP_EVAL --fov_mod $FOVMOD_EVAL --mask_dst $TEST_MASK_FN
python kb_raymap.py --path $DATASET_DIR \
                    --step $STEP_EVAL --fov_mod $FOVMOD_EVAL --gridmap_restrict
# render
python render.py \
    -m $OUTPUT_DIR \
    -s $DATASET_DIR \
    --iteration $ITERS_NUM \
    --camera_model FISHEYE \
    --skip_train \
    --mask_path $DATASET_DIR$FOVMAP_DIR_EVAL$TEST_MASK_FN \
    --raymap_path "$DATASET_DIR"raymap_fisheye.npy \
    --sample_step $STEP_EVAL --fov_mod $FOVMOD_EVAL \
    --train_test_exp \

# wrap back to origianal space
echo "Ground truth (kb) remapping from FoVMap"
python extract_kb.py --path $DATASET_DIR \
                    --src $OUTPUT_DIR/test/ours_$ITERS_NUM/gt \
                    --dst $OUTPUT_DIR/test/ours_$ITERS_NUM/gt_remap \
                    --step $STEP_EVAL --fov_mod $FOVMOD_EVAL --gridmap_restrict

python extract_kb.py --path $DATASET_DIR \
                     --src $OUTPUT_DIR/test/ours_$ITERS_NUM/renders \
                     --dst $OUTPUT_DIR/test/ours_$ITERS_NUM/renders_remap \
                     --step $STEP_EVAL --fov_mod $FOVMOD_EVAL --gridmap_restrict

# evaluation
python metrics.py \
    -m $OUTPUT_DIR \
    --iters $ITERS_NUM \
    --custom_gt /home/Fisheye-GS/output_scannetpp_fs_gt/scannetpp_fs/scannetpp/dslr/$SCENE_ID/test/ours_$ITERS_NUM/gt_remap \
    --block_mask
    #  --use_remap \
    # --custom_gt /home/scannetpp_ever_gt/dslr/$SCENE_ID/test/ours_$ITERS_NUM/gt \
