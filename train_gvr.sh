set -e
# SCENE_IDS="0a5c013435 1f7cbbdde1 1d003b07bd 0a7cc12c0e 2a1a3afad9" done
SCENE_IDS="e3ecd49e2b"
DATA_ROOT="/media/scannetpp/demo/"

STEP=2e-3
FOVMOD_TRAIN=1.3
FOVMOD_EVAL=2.0

ITERS_NUM=30000

for SCENE_ID in $SCENE_IDS; do
    echo "Processing scene: $SCENE_ID"

    DATASET_DIR="$DATA_ROOT$SCENE_ID/dslr/"
    OUTPUT_DIR="./output/scannetpp/$SCENE_ID"

    FOVMAP_DIR_TRAIN="undistorted_fovmaps_fov_${FOVMOD_TRAIN}_step_${STEP}/"
    FOVMAP_DIR_EVAL="undistorted_fovmaps_fov_${FOVMOD_EVAL}_step_${STEP}/"

    TRAIN_MASK_FN="fov_${FOVMOD_TRAIN}_step_${STEP}_mask.png"
    TEST_MASK_FN="fov_${FOVMOD_EVAL}_step_${STEP}_mask.png"

    # Train: Generate fov map
    python prepare_fov.py --path "$DATASET_DIR" --dst "$FOVMAP_DIR_TRAIN" --step "$STEP" --fov_mod "$FOVMOD_TRAIN" --mask_dst "$TRAIN_MASK_FN"

    # Train: Run training script
    python train.py -s "$DATASET_DIR" -m "$OUTPUT_DIR" \
        --iterations "$ITERS_NUM" \
        --checkpoint_iterations 200 300 500 700 1000 2000 3000 4000 7000 8000 9000 10000 12000 15000 17000 20000 22000 25000 27000 30000 \
        --save_iterations 200 300 500 700 1000 2000 3000 4000 7000 8000 9000 10000 12000 15000 17000 20000 22000 25000 27000 30000 \
        --test_iterations 200 300 500 700 1000 2000 3000 4000 7000 8000 9000 10000 12000 15000 17000 20000 22000 25000 27000 30000 \
        --resolution 1 \
        --eval \
        --sample_step "$STEP" --fov_mod "$FOVMOD_TRAIN" \
        --mask_path "${DATASET_DIR}${FOVMAP_DIR_TRAIN}${TRAIN_MASK_FN}" \
        --sibr_mask_refcam "${DATASET_DIR}colmap/cameras_fish.txt"
done
