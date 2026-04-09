from nemo_skills.pipeline.cli import generate, wrap_arguments

model = 'azure/openai/gpt-5.4'
prompt = 'recipes/proof-to-finalanswer/prompts/validity-difficulty.yaml'

generate(
    ctx=wrap_arguments(
        f"++prompt_config={prompt} "
        "++inference.reasoning_effort=high "
        "++server.enable_soft_fail=True "

    ),
    model=f'{model}',
    server_type='openai',
    input_file='/home/eminasyan/workspace/data/validity-difficulty/fa_problems.jsonl',
    server_address="https://inference-api.nvidia.com/v1",
    output_dir='/home/eminasyan/workspace/data/validity-difficulty/results',
)