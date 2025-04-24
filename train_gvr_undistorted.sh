set -e
# SCENE_IDS="1d003b07bd 4ef75031e3"
SCENE_IDS="1d003b07bd 2a1a3afad9 1f7cbbdde1 0a5c013435"
# SCENE_IDS="0a5c013435"
DATA_ROOT="/media/scannetpp/demo/"

STEP_TRAIN=0.002
FOVMOD_TRAIN=1.0

ITERS_NUM=30000

for SCENE_ID in $SCENE_IDS; do
    echo "Processing scene: $SCENE_ID"

    if [ "$SCENE_ID" = "1d003b07bd" ]; then
        STEP_TRAIN=0.002
        FOVMOD_TRAIN=0.85
    fi

    DATASET_DIR="$DATA_ROOT$SCENE_ID/dslr/"
    if [ "$FOVMOD_TRAIN" = "1.0" ]; then
        OUTPUT_DIR="./output_gvr_undistort/scannetpp/$SCENE_ID"
    else
        OUTPUT_DIR="./output_gvr_undistort/scannetpp_fov$FOVMOD_TRAIN/$SCENE_ID"
    fi

    FOVMAP_DIR_TRAIN="undistorted_images/"

    TRAIN_MASK_FN="fov_${FOVMOD_TRAIN}_step_${STEP_TRAIN}_mask.png"

    # Train: Generate fov map
    python prepare_undistorted_pinhole.py --path $DATASET_DIR --dst $FOVMAP_DIR_TRAIN --step $STEP_TRAIN --fov_mod $FOVMOD_TRAIN --mask_dst $TRAIN_MASK_FN

    # Train: Run training script
    python train.py -s "$DATASET_DIR" -m "$OUTPUT_DIR" \
        --iterations "$ITERS_NUM" \
        --checkpoint_iterations 3000 7000 15000 30000 \
        --save_iterations 3000 7000 15000 30000 \
        --test_iterations 200 300 500 700 1000 2000 3000 4000 7000 8000 9000 10000 12000 15000 17000 20000 22000 25000 27000 30000 \
        --resolution 1 \
        --eval \
        --sample_step "$STEP_TRAIN" --fov_mod "$FOVMOD_TRAIN" \
        --mask_path "${DATASET_DIR}${FOVMAP_DIR_TRAIN}${TRAIN_MASK_FN}" \
        # --sibr_mask_refcam "${DATASET_DIR}colmap/cameras_fish.txt"
done
