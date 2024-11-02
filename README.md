# big5-llm-negotiator
This is a negotiation simulation framework incorporating LLM agents with synthetic Big-Five personality traits.

## Setup
1. Install packages with pipenv.
   ```sh
   pipenv sync
   ```
2. Setup an [llm server](https://docs.vllm.ai/en/v0.6.0/serving/openai_compatible_server.html) if needed.

3. Create a config file named `.env` which contains the following lines:
   ```sh
   OPENAI_API_KEY=...
   OPENAI_ORGANIZATION=...
   LLM_SERVER=
   HUGGING_FACE_TOKEN=...
   ```

## Usage
1. Extract agent configs. (each agent will have a randomly assigned personality profile.)
   ```sh
   python agent_profile_generation.py --out_fn $CONFIG_FILE
   ```
2. Generate synthetic negotiation dialogs.
   ```sh
   python bargain_simulation.py --agent_config_fn $CONFIG_FILE --out_dir $RESULT_DIR 
   ```
3. Evaluation & analysis.
   ```sh
   python eval.py --result_dir $RESULT_DIR
   ``` 

## References
- Yin Jou Huang, Rafik Hadfi: How Personality Traits Influence Negotiation Outcomes?  A Simulation based on Large Language Models, Findings of the Association for Computational Linguistics: EMNLP 2024, Miami, USA. (2024.11) [arxiv](https://www.arxiv.org/abs/2407.11549)