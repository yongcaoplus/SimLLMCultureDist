# python construct_sft_data.py --save_type test --save_path ../data/sft_wvs_test.json
# python construct_sft_data.py --save_type train --save_path ../data/sft_wvs_train.json
# python construct_sft_data.py --save_type valid --save_path ../data/sft_wvs_valid.json

# python construct_sft_data.py --save_type test --save_path dataset/sft_wvs_test.json
# python construct_sft_data.py --save_type train --save_path ../data/sft_wvs_train_new.json
# python construct_sft_data.py --save_type valid --save_path ../data/sft_wvs_valid_new.json


python construct_sft_data.py --save_type test \
    --save_path dataset/sft_wvs_test.json \
    --dataset_path dataset/wvs_cn.json