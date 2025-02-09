# Working Qwen 0.5b on GRPO: Training a Small Math Reasoner with RL

This Colab notebook ([Open In Colab](https://colab.research.google.com/github/HarleyCoops/OneShotGRPO/blob/main/PublicWorkingGRPO.ipynb)) demonstrates how to train a small language model (Qwen 0.5b) on a math dataset (GSM8K) using Reinforcement Learning (RL), specifically the Generative Reinforcement Policy Optimization (GRPO) algorithm.  It's a more comprehensive and functional version of the [GRPO demo](https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb) by Will Brown, addressing some of the common difficulties in setting up a complete GRPO training pipeline.

This version has two parts: the Colab notebook and the single python script here in the repo. You need to connect the Collab notebook to a single A100. This took about 60 compute units which is at most $7.50 USD depending upon your subscription. 

After you run the collab notebook, just use the python script to call back to your model name and that is it. You are running inference in your own model in about 3 hours for about $10; all complete with a full Weights and Biases dataset along with a Hugging Face deployment and almost free inference. 

You can even use this as a cheap way to generate your own sythetic datasets as the python script saves down a JSONL of your query and all the reasoning. 


**Key Features and Goals:**

*   **Trains a Model:** Fine-tunes the Qwen 0.5b model on the GSM8K dataset using GRPO.
*   **Downloads the Model:** Provides instructions and code to download the trained model weights.
*   **Publishes to Hugging Face:** Includes steps to upload the trained model to your Hugging Face account.
*   **Local Inference:** Shows how to run inference in Python using the locally trained model, with results returned to the terminal.
*   **Reasoning Trace:**  The model is trained to output its reasoning steps in a structured XML format, allowing for better understanding and debugging of its problem-solving process.
* **Clear Explanations**: Includes in-depth, markdown documentation explaining the core components, data processing, and rationale behind the hyperparameter choices.

**Project Structure and Workflow**
The notebook guides the user through the following main steps.

1.  **Setup and Dependencies:**
    *   Install vLLM.
    *   Install TRL and the Datasets library.
    *   Import the core libraries.

2.  **Data Loading and Preprocessing:**
    *   **Dataset:** Uses the GSM8K (Grade School Math 8K) dataset from Hugging Face.
    *   **Formatting:** Defines a specific response format (`SYSTEM_PROMPT` and `XML_COT_FORMAT`) to encourage the model to output its reasoning process in a structured way (chain-of-thought prompting).
    *   **Extraction Functions:** Includes functions (`extract_xml_answer`, `extract_hash_answer`) to parse the model's output and the dataset's answer format.
    *   **Data Loading Function:**  The `get_gsm8k_questions` function loads and preprocesses the GSM8K dataset, adding the system prompt and formatting the data for training.

3.  **Reward Functions:**
    *   Defines multiple reward functions to guide the RL training process:
        *   `correctness_reward_func`:  Checks if the extracted answer matches the ground truth. (Primary reward)
        *   `int_reward_func`:  Ensures the answer is a numerical value.
        *   `strict_format_reward_func`:  Enforces the exact XML format (newlines, tags).
        *   `soft_format_reward_func`:  Allows for more flexible whitespace within the XML structure.
        *   `xmlcount_reward_func`:  Scores the individual XML components (opening/closing tags).
    *   The combination of these rewards encourages the model to produce correct answers *and* articulate its reasoning in a consistent, parsable format.

4.  **Model and Training Configuration:**
    *   **Model:** Uses the `Qwen/Qwen2.5-0.5B-Instruct` model from Hugging Face.
    *   **vLLM Integration:** Uses vLLM for efficient inference, significantly speeding up the training process.  A specific `gpu_memory_utilization` is set for vLLM to balance memory usage between training and inference.
    *   **TRL (Transformer Reinforcement Learning):** Leverages the `trl` library for GRPO training.
        *   `GRPOConfig`:  Defines the training hyperparameters, including:
            *   `learning_rate`: Small learning rate (5e-6) appropriate for fine-tuning LLMs.
            *   `adam_beta1`, `adam_beta2`:  Adam optimizer parameters (0.9, 0.99).
            *   `lr_scheduler_type`: 'cosine' for smooth learning rate decay.
            *   `bf16`:  Uses bfloat16 precision for efficient computation.
            *   `per_device_train_batch_size`: Batch size of 1 per device.
            *   `gradient_accumulation_steps`: Accumulates gradients over 4 steps to simulate a larger batch size.
            *   `num_generations`: Generates 16 responses per prompt for exploration.
            *   `max_prompt_length`, `max_completion_length`:  Control input and output lengths.
            *   `num_train_epochs`:  Trains for only 1 epoch to prevent overfitting on the relatively small GSM8K dataset.
            *   `save_steps`:  Saves checkpoints every 100 steps.
            *   `max_grad_norm`:  Clips gradients to prevent instability.
            *    `report_to`: 'wandb', so that experiments are tracked using wandb
        *   `GRPOTrainer`:  The core training engine that handles the RL loop, reward computation, and policy updates.

5.  **Training Execution:**
    *   The `trainer.train()` command initiates the training process.
    *   The notebook includes detailed output showing the training progress, including the questions, expected answers, model responses, and extracted answers.

6.  **Saving and Loading the Model:**
    *   Provides code to save the model weights to a local zip file.
    *   Demonstrates how to load the trained model (both architecture and weights) from the checkpoint directory.

7. **Push to Hugging Face Hub:**
    *   Login using the hugging face cli
    *   Push model weights, tokenizer, and configs to your account.

8. **Model Inference**
     *  A section of code that allows users to easily test their trained model with new questions.

## In-Depth Explanations

The notebook provides detailed explanations for many of the design choices and technical aspects, including:

*   **vLLM:** Explains the benefits of vLLM for efficient LLM inference and its role in the project.
*   **TRL and Datasets:**  Describes how TRL and the Hugging Face Datasets library are used for RL training.
*   **GRPOConfig Parameters:**  Provides a deep dive into the rationale behind each training argument, linking them to relevant research and theoretical concepts. This is particularly important for understanding how the hyperparameters affect the RL training process.
*   **Reward Functions:** Explains the purpose and implementation of each reward function, emphasizing how they work together to shape the model's behavior.
*   **Data Processing:**  Details the steps involved in preparing the GSM8K dataset for training, including formatting, instruction integration, and answer extraction.
* **Checkpoints** Provides a detailed look at the files in the model's checkpoint.

## Getting Started

1.  **Open in Colab:** Click the "Open In Colab" badge at the top of this README to run the notebook in Google Colab.
2.  **Install Dependencies:**  Run the `!pip install` cells to install the required libraries (vLLM, trl, datasets, wandb).  Remember to restart the runtime after installing vLLM, as noted in the notebook.
3.  **Set Up Weights & Biases (optional):**  If you want to use Weights & Biases for experiment tracking, run the `wandb.login()` cell and follow the instructions to authenticate.
4.  **Run the Training Cells:** Execute the code cells sequentially to load the data, configure the model and trainer, and start the training process.
5.  **Test Inference:** Use the inference code at the end to test the trained model with your own math problems.
6.  **Download Checkpoint:** Download the model checkpoint as a zip file for later use.
7. **Upload Model:** Upload your fine-tuned model to your own hugging face account.

## Notes and Considerations

*   **Runtime:**  Training can take several hours, even on a GPU.  The exact time will depend on the hardware and the specific training parameters.
*   **Overfitting:**  Because the notebook trains for only one epoch on a relatively small dataset, the model might not generalize perfectly to all types of math problems.  More extensive training might be necessary for better performance.
*   **Reward Tuning:**  The reward functions and their weights are crucial for shaping the model's behavior.  Experimenting with different reward functions and weights can lead to improved results.
*   **Hyperparameter Optimization:** The training arguments in `GRPOConfig` can be further tuned to optimize performance.  Consider using a hyperparameter search library like Optuna or Ray Tune.
* **PEFT** The use of PEFT is described as having risk, but could be useful for some users.
