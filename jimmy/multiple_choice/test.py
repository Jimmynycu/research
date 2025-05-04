# tag: bigbench_multiple_choice_a
# dataset_path: hails/bigbench
# dataset_kwargs:
#   # num_shots: 0 # TODO: num of shots for `bigbench` HF dataset should be controlled through this, not through the typical methods
#   # subtask_name: null
# output_type: multiple_choice
# test_split: default
# doc_to_text: "{{inputs}}"
# doc_to_target: "{{multiple_choice_targets.index(targets[0])}}"
# doc_to_choice: "{{multiple_choice_targets}}"
# metric_list:
#   - metric: acc
#   # TODO: brier score and other metrics
# metadata:
#   version: 1.0

# write a code that can generate a yaml file with each is define in list

import yaml
class LiteralTag(str):
    pass

def literal_tag_representer(dumper, data):
    # Represent the value as a YAML tag without quotes
    return dumper.represent_scalar('!function', data, style=None)

yaml.add_representer(LiteralTag, literal_tag_representer)

# Shared configuration for YAML files
bigbench_abstract_config = {
    "dataset_name": "abstract_narrative_understanding_zero_shot",
    "tag": "custom",
    "dataset_path": "hails/bigbench",
    "dataset_kwargs": {
        # num_shots: 0 # TODO: num of shots for `bigbench` HF dataset should be controlled through this, not through the typical methods
        # subtask_name: null
    },
    "output_type": "multiple_choice",
    "test_split": "default",
    "doc_to_target": "{{multiple_choice_targets.index(targets[0])}}",
    "doc_to_choice": "{{multiple_choice_targets}}",
    "metric_list": [
        {"metric": "acc"}
        # TODO: brier score and other metrics
    ],
    "metadata": {"version": 1.0},
}
# Configuration for HellaSwag
hellaswag_config = {
    "tag": "multiple_choice",
    "task": "hellaswag",
    "dataset_path": "hellaswag",
    "dataset_name": None,
    "output_type": "multiple_choice",
    "training_split": "train",
    "validation_split": "validation",
    "test_split": None,
    "process_docs": LiteralTag("utils.process_docs"),  # Use LiteralTag here
    "doc_to_text": "{{query}}",
    "doc_to_target": "{{label}}",
    "doc_to_choice": "choices",
    "metric_list": [
        {"metric": "acc", "aggregation": "mean", "higher_is_better": True},
        {"metric": "acc_norm", "aggregation": "mean", "higher_is_better": True}
    ],
    "metadata": {"version": 1.0},
    "dataset_kwargs": {
        "trust_remote_code": True
    },
}

# Unique configurations for each YAML file
tasks = [
    "hellaswag_a",
    "hellaswag_b",
    "hellaswag_c",
    "hellaswag_d",
    "hellaswag_e",
    "hellaswag_f",
    "hellaswag_g"
]

doc_to_texts = [
    "\"Write your answer and give me a confidence score between 0-1 for your answer. {{query}}\"",
    "\"This is very important to my career. {{query}}\"",
    "\"You'd better be sure. {{query}}\"",
    "\"Are you sure? {{query}}\"",
    "\"Are you sure that's your final answer? It might be worth taking another look. {{query}}\"",
    "\"{{query}}\"",
    "\"{{query}}\""
]

output_files = [
    "template1.yaml",
    "template2.yaml",
    "template3.yaml",
    "template4.yaml",
    "template5.yaml",
    "template6.yaml",
    "template7.yaml"
]

def generate_yaml_files():
    for i in range(len(output_files)):
        config = {
            **hellaswag_config,
            "task": tasks[i],  # Ensure task is a single string, not a list
            "doc_to_text": doc_to_texts[i],
            "process_docs": "!function utils.process_docs"
        }
        with open(output_files[i], "w") as file:
            yaml.dump(config, file, default_flow_style=False, sort_keys=False)
        print(f"YAML file '{output_files[i]}' has been created.")

def main() -> None:
    generate_yaml_files()

if __name__ == "__main__":
    main()