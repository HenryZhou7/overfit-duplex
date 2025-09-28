# overfit-duplex

## Env set up

Dev environment:
```
# Get the relevant codes
git submodule init && git submodule update
# Initialize the necesary environment variables.
touch .env
# Sync the environment
uv sync
```

Pre-trained weights:
```
tune download meta-llama/Llama-3.2-1B-Instruct --output-dir $HOME/model_weights/Llama-3.2-1B-Instruct --ignore-patterns "original/consolidated.00.pth" --hf-token <HF_TOKEN>
```

## Coding Convention

1. Assume the libraries are already included. No need to do `try - except` block on imports.
