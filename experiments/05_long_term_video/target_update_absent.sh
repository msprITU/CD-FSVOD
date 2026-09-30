#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# the dataloader needs the VOT class list and takes the query from the first frame
ipython "$SCRIPT_DIR/../../tools/configure_dataloader.ipy" VOT 0
cd /root/BHRL


videos=("carchase" "following" "horseride" "liverRun" "volkswagen" "longboard")

if [ $# -gt 0 ]; then
    videos=("$@")
fi



for seq in "${videos[@]}"; do
    if [ ! -d "vot_results/target_update_studies/real_time/e100_IoU_0_7_parts/parts_100/coco/logs/${seq}" ]; then
        mkdir -p "vot_results/target_update_studies/real_time/e100_IoU_0_7_parts/parts_100/coco/logs/${seq}"
    fi
    if [ ! -d "work_dirs/vot/BHRL/model_coco/${seq}" ]; then
        mkdir -p "work_dirs/vot/BHRL/model_coco/${seq}"
    fi
    python tools/train.py \
      --resume_from checkpoints/model_split4.pth \
      --config configs/vot/BHRL.py \
      --seq_name ${seq} \
      --no-validate \
      --seq_img_id 0 \
      --requested_class 20 \
      --ann_file vot_annotation/ft/${seq}_first_ft.json \
      --work_dir work_dirs/vot/BHRL/model_coco/${seq}

    python scripts/split_seq_imgs.py ${seq} 100 > /dev/null
    # index of the last 100-frame part of the full timeline (object-absent frames included)
    seq_parts_num=$(( $(ls vot_annotation/${seq}/with_absent/${seq}_part_*.json | wc -l) - 1 ))
    python tools/test.py \
        --config configs/vot/BHRL.py \
        --seq_name ${seq} \
        --part 0 \
        --ann_file vot_annotation/${seq}/with_absent/${seq}_part_0.json \
        --checkpoint work_dirs/vot/BHRL/model_coco/${seq}/epoch_109.pth \
        --eval bbox \
        --result_file "vot_results/target_update_studies/real_time/e100_IoU_0_7_parts/parts_100/coco/results/${seq}" | tee vot_results/target_update_studies/real_time/e100_IoU_0_7_parts/parts_100/coco/logs/${seq}/${seq}_part0.out

    python scripts/find_update_frame_absent.py ${seq} 0 100 coco | tee vot_results/target_update_studies/real_time/e100_IoU_0_7_parts/parts_100/coco/logs/${seq}/${seq}_update_frames_0.out
    echo "The variable value is: $seq_parts_num"
    for i in $(seq 1 $seq_parts_num); do
        log_file="vot_results/target_update_studies/real_time/e100_IoU_0_7_parts/parts_100/coco/logs/${seq}/${seq}_part${i}.out"

        python tools/train.py \
            --resume_from work_dirs/vot/BHRL/model_coco/${seq}/epoch_109.pth \
            --config configs/vot/BHRL.py \
            --seq_name ${seq} \
            --no-validate \
            --seq_img_id 0 \
            --requested_class 20 \
            --ann_file vot_results/target_update_studies/real_time/e100_IoU_0_7_parts/parts_100/coco/update_frames/${seq}/target_update.json \
            --work_dir work_dirs/vot/BHRL/target_updates/${seq}


        python tools/test.py \
            --config configs/vot/BHRL.py \
            --seq_name ${seq} \
            --part $i \
            --ann_file vot_annotation/${seq}/with_absent/${seq}_part_$i.json \
            --checkpoint work_dirs/vot/BHRL/target_updates/${seq}/epoch_209.pth \
            --result_file "vot_results/target_update_studies/real_time/e100_IoU_0_7_parts/parts_100/coco/results/${seq}" \
            --eval bbox | tee $log_file

        # the last part has no following part to prepare a target update for
        if [ $i -lt $seq_parts_num ]; then
            python scripts/find_update_frame_absent.py ${seq} $i 100 coco | tee vot_results/target_update_studies/real_time/e100_IoU_0_7_parts/parts_100/coco/logs/${seq}/${seq}_update_frames_$i.out
        fi
    done
done

