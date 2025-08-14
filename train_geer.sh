set -e
# DATA_ROOT="/media/scannetpp/demo/"
# SCENE_IDS="0a5c013435 1d003b07bd 4ef75031e3 2a1a3afad9 1f7cbbdde1"
# DATA_ROOT="/media/projectaria_tools_aria-scenes_data/scannetpp_formatted/"
# SCENE_IDS="steakhouse_patio"
DATA_ROOT="/media/bosch_handhold_device/scannetpp_formatted/"
SCENE_IDS="bosch_miniPoC"

STEP=0.002
FOVMOD_TRAIN=1.0

ITERS_NUM=12000

for SCENE_ID in $SCENE_IDS; do
    echo "Processing scene: $SCENE_ID"

    # DATASET_DIR="$DATA_ROOT$SCENE_ID/dslr/"
    DATASET_DIR="$DATA_ROOT$SCENE_ID/"
    OUTPUT_DIR="./output/$SCENE_ID"

    FOVMAP_DIR_TRAIN="undistorted_fovmaps_fov_${FOVMOD_TRAIN}_step_${STEP}/"

    TRAIN_MASK_FN="fov_${FOVMOD_TRAIN}_step_${STEP}_mask.png"

    # # Train: Generate fov map
    # python prepare_fov.py --path "$DATASET_DIR" --dst "$FOVMAP_DIR_TRAIN" --step "$STEP" --fov_mod "$FOVMOD_TRAIN" --mask_dst "$TRAIN_MASK_FN"

    # # Train: Run training script
    # python train.py -s "$DATASET_DIR" -m "$OUTPUT_DIR" \
    #     --iterations "$ITERS_NUM" \
    #     --checkpoint_iterations 7000 12000 \
    #     --save_iterations 7000 12000 \
    #     --test_iterations 200 300 500 700 1000 2000 3000 4000 7000 8000 9000 10000 12000 \
    #     --resolution 1 \
    #     --eval \
    #     --sample_step "$STEP" --fov_mod "$FOVMOD_TRAIN" \
    #     --sibr_mask_refcam "${DATASET_DIR}colmap/cameras_fish.txt" \
    #     --mask_path "${DATASET_DIR}${FOVMAP_DIR_TRAIN}${TRAIN_MASK_FN}" \
    #     # --mask_path "${DATASET_DIR}${FOVMAP_DIR_TRAIN}/merged.png" \

    # render
    python render.py \
        -m $OUTPUT_DIR \
        -s $DATASET_DIR \
        --iteration $ITERS_NUM \
        --camera_model FISHEYE \
        --skip_train \
        --mask_path "${DATASET_DIR}${FOVMAP_DIR_TRAIN}${TRAIN_MASK_FN}" \
        --sample_step $STEP --fov_mod $FOVMOD_TRAIN \
        --train_test_exp \
        # --add_ego_mask
done
