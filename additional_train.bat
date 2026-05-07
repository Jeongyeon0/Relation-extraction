python train_candidate_scoring_dot+contrastive+learnable+temperature.py ^
  --mode train ^
  --train_path refined-klue-re/train_split.tsv ^
  --dev_path refined-klue-re/valid_split.tsv ^
  --model_name klue/roberta-base ^
  --output_dir outputs/re_klue_roberta_base_mlp_dot_contrastive_learnable_temp ^
  --max_length 512 ^
  --batch_size 16 ^
  --eval_batch_size 32 ^
  --epochs 10 ^
  --learning_rate 2e-5 ^
  --init_temperature 0.07 ^
  --fp16



python train_candidate_scoring_dot+contrastive+learnable+temperature.py ^
  --mode train ^
  --train_path refined-klue-re/train_split.tsv ^
  --dev_path refined-klue-re/valid_split.tsv ^
  --model_name klue/roberta-base ^
  --output_dir outputs/re_klue_roberta_base_mlp_dot_contrastive_learnable_temp_descript_short(korean) ^
  --label_desc_path label_descript.json ^
  --max_length 512 ^
  --batch_size 16 ^
  --eval_batch_size 32 ^
  --epochs 10 ^
  --learning_rate 2e-5 ^
  --init_temperature 0.07 ^
  --fp16


python train_candidate_scoring_mlp+dot+contrastive+learnable+temperature.py ^
  --mode train ^
  --train_path refined-klue-re/train_split.tsv ^
  --dev_path refined-klue-re/valid_split.tsv ^
  --model_name klue/roberta-base ^
  --output_dir outputs/re_klue_roberta_base_mlp_dot_contrastive_learnable_temp ^
  --max_length 512 ^
  --batch_size 16 ^
  --eval_batch_size 32 ^
  --epochs 10 ^
  --learning_rate 2e-5 ^
  --init_temperature 0.07 ^
  --fp16



python train_candidate_scoring_mlp+dot+contrastive+learnable+temperature.py ^
  --mode train ^
  --train_path refined-klue-re/train_split.tsv ^
  --dev_path refined-klue-re/valid_split.tsv ^
  --model_name klue/roberta-base ^
  --output_dir outputs/re_klue_roberta_base_mlp_dot_contrastive_learnable_temp_descript_short(korean) ^
  --label_desc_path label_descript.json ^
  --max_length 512 ^
  --batch_size 16 ^
  --eval_batch_size 32 ^
  --epochs 10 ^
  --learning_rate 2e-5 ^
  --init_temperature 0.07 ^
  --fp16
