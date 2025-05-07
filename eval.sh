#!/bin/bash

# Function to login to Hugging Face
huggingface-cli login --token

cd new

# Define the model path and result path variables
models=(
    "meta-llama/Llama-3.2-3B"
    # "Jimmy1229/Llama-3.2-3B-4bit"
    # "Jimmy1229/Llama-3.2-3B-8bit"
    # "meta-llama/Llama-3.2-1B"
    # "Jimmy1229/Llama-3.2-1B-4bit"
    # "Jimmy1229/Llama-3.2-1B-8bit"
    # "simplescaling/s1.1-3B"
    # "Jimmy1229/s1.1-3B-8bit-GPTQ"
    # "Jimmy1229/s1.1-3B-4bit-GPTQ"
    # "Jimmy1229/s1.1-1.5B-4bit-GPTQ"
    # "Jimmy1229/s1.1-1.5B-8bit-GPTQ"
    # "simplescaling/s1.1-1.5B"
    # simplescaling/s1.1-32B
    # "microsoft/phi-4"
)
results=(
    "Llama-3.2-3B"
    # "Llama-3.2-3B-4bit"
    # "Llama-3.2-3B-8bit"
    # "Llama-3.2-1B"
    # "Llama-3.2-1B-4bit"
    # "Llama-3.2-1B-8bit"
    # "s1.1-3B"
    # "s1.1-3B-8bit-GPTQ"
    # "s1.1-3B-4bit-GPTQ"
    # "s1.1-1.5B-4bit-GPTQ"
    # "s1.1-1.5B-8bit-GPTQ"
    # "s1.1-1.5B"
    # "s1.1-1.5B"
    # "microsoft/phi-4"
)

# task=$(echo "bigbench_abstract_narrative_understanding_multiple_choice,"\
#     "bigbench_english_proverbs_multiple_choice,"\
#     "bigbench_general_knowledge_multiple_choice,"\
#     "bigbench_logic_grid_puzzle_multiple_choice,"\
#     "bigbench_mathematical_induction_multiple_choice,"\
#     "bigbench_physical_intuition_multiple_choice,"\
#     "bigbench_riddle_sense_multiple_choice,"\
#     "bigbench_what_is_the_tao_multiple_choice,"\
#     "bigbench_winowhy_multiple_choice,"\
#     "bigbench_logic_grid_puzzle_multiple_choice,"\
#     "commonsense_qa,"\
#     "arc_challenge,"\
#     "logiqa,"\
#     "mathqa,"\
#     "agieval_logiqa_en,"\
#     "global_mmlu_full_en_college_mathematics,"\
#     "mmlu_college_mathematics,"\
#     "global_mmlu_en_humanities,"\
#     "global_mmlu_full_en_abstract_algebra,"\
#     "global_mmlu_full_en_anatomy,"\
#     "global_mmlu_full_en_astronomy,"\
#     "global_mmlu_full_en_business_ethics,"\
#     "global_mmlu_full_en_clinical_knowledge,"\
#     "global_mmlu_full_en_college_biology,"\
#     "global_mmlu_full_en_college_chemistry,"\
#     "global_mmlu_full_en_college_computer_science,"\
#     "global_mmlu_full_en_college_medicine,"\
#     "global_mmlu_full_en_college_physics,"\
#     "mmlu_business_ethics,"\
#     "mmlu_clinical_knowledge,"\
#     "mmlu_college_biology,"\
#     "mmlu_college_chemistry,"\
#     "mmlu_college_computer_science,"\
#     "mmlu_college_mathematics,"\
#     "mmlu_college_medicine,"\
#     "mmlu_college_physics,"\
#     "mmlu_computer_security,"\
#     "mmlu_conceptual_physics,"\
#     "mmlu_continuation_abstract_algebra,"\
#     "mmlu_continuation_anatomy,"\
#     "mmlu_continuation_astronomy,"\
#     "mmlu_continuation_business_ethics,"\
#     "mmlu_continuation_clinical_knowledge,"\
#     "mmlu_continuation_college_biology,"\
#     "mmlu_continuation_college_chemistry,"\
#     "mmlu_continuation_college_computer_science,"\
#     "mmlu_continuation_college_mathematics,"\
#     "mmlu_continuation_college_medicine,"\
#     "mmlu_continuation_college_physics,"\
#     "cola,"\
#     "agieval_sat_math,"\
#     "blimp_anaphor_gender_agreement,"\
#     "copa,"\
#     "ethics_utilitarianism,"\
#     "anli_r1,"\
#     "truthfulqa_mc1,"\
#     "ethics_justice" | tr -d ' '
# )

task="hellaswag_5"

# Run lm_eval for each model and its corresponding result
# for i in "${!models[@]}"; do
#     model_path="${models[$i]}"
#     result_path="${results[$i]}"
    
#     lm_eval --model hf \
#         --model_args pretrained=$model_path,parallelize=True,trust_remote_code=True,do_sample=True \
#         --tasks $task \
#         --write_out \
#         --device cuda:6 \
#         --output_path $result_path \
#         --log_samples \
#         --verbosity DEBUG
# done


for i in "${!models[@]}"; do
    model_path="${models[$i]}"
    result_path="${results[$i]}"

    accelerate launch -m lm_eval --model hf \
        --model_args pretrained=$model_path,parallelize=True,trust_remote_code=True,do_sample=True\
        --tasks $task \
        --batch_size 16 \
        --write_out \
        --output_path $result_path \
        --log_samples \
        --device cuda:4 \
        --verbosity DEBUG

done
