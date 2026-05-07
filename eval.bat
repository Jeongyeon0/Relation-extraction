python train_candidate_scoring_mlp+bilinear.py ^
  --mode eval ^
  --eval_path refined-klue-re/dev.tsv ^
  --eval_model_dir outputs/re_klue_roberta_base_mlp_bilinear/best_model ^
  --output_dir outputs/re_klue_roberta_base_mlp_bilinear ^
  --eval_output_name dev_eval_predictions.tsv ^
  --max_length 512 ^
  --eval_batch_size 32