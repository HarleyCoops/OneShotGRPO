# Gradio Deployment Guide

##  Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Building Chat Interfaces](#building-chat-interfaces)
- [HuggingFace Integration](#huggingface-integration)
- [Deployment Options](#deployment-options)
- [Advanced Features](#advanced-features)
- [Production Considerations](#production-considerations)
- [Troubleshooting](#troubleshooting)

---

## Overview

Gradio makes it easy to create beautiful web interfaces for machine learning models. This guide shows you how to build and deploy chat interfaces for GRPO-trained models.

### Key Features

- ** Fast Development**: Create UIs in minutes
- ** Beautiful Defaults**: Professional look out of the box
- ** Mobile Responsive**: Works on all devices
- ** Easy Sharing**: Public URLs with one parameter
- ** HF Integration**: Deploy to Spaces with one click
- ** Streaming**: Real-time response generation

---

## Quick Start

### Installation

```python
!pip install -q gradio
import gradio as gr
```

### Minimal Chat Interface

```python
def chat(message, history):
    """Simple echo chatbot"""
    return f"You said: {message}"

# Create interface
demo = gr.ChatInterface(fn=chat)

# Launch
demo.launch()
```

### With GRPO Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "your-username/grpo-math-model",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("your-username/grpo-math-model")

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def solve_math(message, history):
    """Solve math problem with GRPO model"""
    # Create prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": message}
    ]

    # Generate
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True
    )

    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract answer
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()

    return response

# Create chat interface
demo = gr.ChatInterface(
    fn=solve_math,
    title=" GRPO Math Tutor",
    description="Ask me grade-school math questions!",
    examples=[
        "If John has 5 apples and buys 3 more, how many does he have?",
        "A rectangle is 4m wide and 6m long. What is its area?",
        "If a pizza is cut into 8 slices and you eat 3, what fraction is left?"
    ]
)

demo.launch(share=True)  # Creates public URL
```

---

## Building Chat Interfaces

### Basic ChatInterface

```python
import gradio as gr

def chat_fn(message, history):
    """
    Args:
        message: Current user message (string)
        history: List of [user_msg, bot_msg] pairs

    Returns:
        Bot response (string)
    """
    return f"Response to: {message}"

gr.ChatInterface(fn=chat_fn).launch()
```

### With State and Parameters

```python
def chat_with_params(message, history, temperature, max_length):
    """Chat with adjustable parameters"""
    # Generate response using parameters
    response = generate_response(
        message,
        temperature=temperature,
        max_length=max_length
    )
    return response

demo = gr.ChatInterface(
    fn=chat_with_params,
    additional_inputs=[
        gr.Slider(0.1, 2.0, value=0.7, label="Temperature"),
        gr.Slider(50, 500, value=200, label="Max Length")
    ]
)

demo.launch()
```

### Streaming Responses

```python
def chat_stream(message, history):
    """Stream response token by token"""
    response = ""

    # Generate tokens
    for token in generate_tokens(message):
        response += token
        yield response  # Yield partial response

# Enable streaming
demo = gr.ChatInterface(
    fn=chat_stream,
    type="messages"  # Required for streaming
)

demo.launch()
```

### Formatted Output

```python
def chat_formatted(message, history):
    """Return formatted markdown response"""
    result = solve_math_problem(message)

    # Format with markdown
    response = f"""
** Reasoning:**

{result['reasoning']}

** Answer:** {result['answer']}
"""

    return response

demo = gr.ChatInterface(fn=chat_formatted)
demo.launch()
```

---

## HuggingFace Integration

### Load from HF Model

```python
import gradio as gr

# Quick method - automatically loads model
demo = gr.load_chat(
    base_url="https://api-inference.huggingface.co/models/",
    model="your-username/grpo-math-model",
    token="hf_YOUR_TOKEN"  # If private
)

demo.launch()
```

### Custom HF Integration

```python
from huggingface_hub import InferenceClient

client = InferenceClient(
    model="your-username/grpo-math-model",
    token="hf_YOUR_TOKEN"
)

def chat_with_hf(message, history):
    """Use HF Inference API"""
    messages = [{"role": "user", "content": message}]

    response = ""
    for chunk in client.chat.completions.create(
        messages=messages,
        stream=True,
        max_tokens=200
    ):
        if chunk.choices[0].delta.content:
            response += chunk.choices[0].delta.content
            yield response

demo = gr.ChatInterface(fn=chat_with_hf)
demo.launch()
```

### Use HF Inference Endpoint

```python
client = InferenceClient(
    base_url="https://your-endpoint.endpoints.huggingface.cloud/v1/",
    token="YOUR_ENDPOINT_TOKEN"
)

def chat_endpoint(message, history):
    """Use dedicated HF Inference Endpoint"""
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message}
        ],
        max_tokens=200,
        temperature=0.7
    )

    return response.choices[0].message.content

demo = gr.ChatInterface(fn=chat_endpoint)
demo.launch()
```

---

## Deployment Options

### 1. Local Development

```python
# Default: localhost:7860
demo.launch()

# Custom port
demo.launch(server_port=8080)

# Show errors in UI
demo.launch(debug=True)
```

### 2. Public URL (Temporary)

```python
# Creates temporary public URL (72 hours)
demo.launch(share=True)

# Output:
# Running on local URL:  http://127.0.0.1:7860
# Running on public URL: https://abc123.gradio.live
```

### 3. HuggingFace Spaces (Permanent)

#### Step 1: Create Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Choose:
   - Space name: `grpo-math-tutor`
   - SDK: Gradio
   - Hardware: CPU (upgrade if needed)

#### Step 2: Create `app.py`

```python
# app.py
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model from HF Hub
model = AutoModelForCausalLM.from_pretrained(
    "your-username/grpo-math-model",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("your-username/grpo-math-model")

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def solve_math(message, history, temperature=0.7):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": message}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        inputs,
        max_new_tokens=200,
        temperature=temperature,
        do_sample=True
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract assistant response
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()

    # Parse and format
    try:
        reasoning = response.split("<reasoning>")[1].split("</reasoning>")[0].strip()
        answer = response.split("<answer>")[1].split("</answer>")[0].strip()
        formatted = f"**Reasoning:**\n\n{reasoning}\n\n**Answer:** {answer}"
    except:
        formatted = response

    return formatted

demo = gr.ChatInterface(
    fn=solve_math,
    title=" GRPO Math Tutor",
    description="Ask me grade-school math problems! I'll show my step-by-step reasoning.",
    examples=[
        "If a pizza is cut into 8 slices and you eat 3, what fraction is left?",
        "A train travels 60 mph for 2.5 hours. How far does it go?",
        "Sarah has $20. She buys 3 books at $4 each. How much money is left?"
    ],
    additional_inputs=[
        gr.Slider(0.1, 1.5, value=0.7, label="Temperature", step=0.1)
    ],
    theme=gr.themes.Soft(),
    retry_btn=" Retry",
    undo_btn="↩ Undo",
    clear_btn=" Clear"
)

if __name__ == "__main__":
    demo.launch()
```

#### Step 3: Create `requirements.txt`

```txt
gradio
transformers
torch
accelerate
```

#### Step 4: Push to Space

```bash
# Clone space
git clone https://huggingface.co/spaces/your-username/grpo-math-tutor
cd grpo-math-tutor

# Add files
cp app.py .
cp requirements.txt .

# Commit and push
git add app.py requirements.txt
git commit -m "Initial commit"
git push
```

#### Step 5: Configure Space

In Space settings:
- **Hardware**: Upgrade to GPU if needed (T4, A10G, A100)
- **Secrets**: Add `HF_TOKEN` if using private models
- **Sleep timeout**: Set to "Never" for always-on

### 4. Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app.py .

# Expose port
EXPOSE 7860

# Run app
CMD ["python", "app.py"]
```

Build and run:

```bash
docker build -t grpo-math-tutor .
docker run -p 7860:7860 grpo-math-tutor
```

---

## Advanced Features

### Custom Themes

```python
from gradio.themes import Soft, Base

# Use built-in theme
demo = gr.ChatInterface(fn=chat, theme=gr.themes.Soft())

# Customize theme
custom_theme = Soft(
    primary_hue="blue",
    secondary_hue="gray",
    font=["Helvetica", "sans-serif"]
)

demo = gr.ChatInterface(fn=chat, theme=custom_theme)
```

### Multi-Modal Interface

```python
def chat_with_image(message, history):
    """Chat that can handle images (future extension)"""
    # Parse message for images
    # Generate response considering images
    return response

demo = gr.ChatInterface(fn=chat_with_image)
```

### Custom Components

```python
with gr.Blocks() as demo:
    gr.Markdown("#  GRPO Math Tutor")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=500)
            msg = gr.Textbox(
                placeholder="Ask a math question...",
                show_label=False
            )

            with gr.Row():
                submit = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear")

        with gr.Column(scale=1):
            gr.Markdown("### Settings")
            temperature = gr.Slider(0.1, 2.0, value=0.7, label="Temperature")
            max_length = gr.Slider(50, 500, value=200, label="Max Length")

            gr.Markdown("### Examples")
            examples = gr.Examples(
                examples=[
                    "What is 25 * 4?",
                    "If x + 5 = 12, what is x?"
                ],
                inputs=msg
            )

    def respond(message, chat_history):
        response = generate_response(message)
        chat_history.append((message, response))
        return "", chat_history

    submit.click(respond, [msg, chatbot], [msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()
```

### Analytics and Logging

```python
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def chat_with_logging(message, history):
    """Chat with analytics"""
    logger.info(f"User query: {message}")

    response = generate_response(message)

    logger.info(f"Bot response length: {len(response)}")

    # Track to analytics service
    track_event("chat_message", {
        "query_length": len(message),
        "response_length": len(response)
    })

    return response

demo = gr.ChatInterface(fn=chat_with_logging)
```

### Rate Limiting

```python
from functools import wraps
import time

# Simple rate limiter
last_request = {}

def rate_limit(max_per_minute=10):
    def decorator(fn):
        @wraps(fn)
        def wrapper(message, history, request: gr.Request):
            client_ip = request.client.host
            now = time.time()

            # Check rate
            if client_ip in last_request:
                elapsed = now - last_request[client_ip]
                if elapsed < 60 / max_per_minute:
                    return "⏱ Please wait a moment before sending another message."

            last_request[client_ip] = now
            return fn(message, history)

        return wrapper
    return decorator

@rate_limit(max_per_minute=5)
def chat_limited(message, history, request: gr.Request):
    return generate_response(message)

demo = gr.ChatInterface(fn=chat_limited)
```

---

## Production Considerations

### 1. Error Handling

```python
def chat_safe(message, history):
    """Chat with error handling"""
    try:
        response = generate_response(message)
        return response
    except torch.cuda.OutOfMemoryError:
        return " GPU out of memory. Please try a shorter question."
    except Exception as e:
        logger.error(f"Error: {e}")
        return " Sorry, something went wrong. Please try again."

demo = gr.ChatInterface(fn=chat_safe)
```

### 2. Input Validation

```python
def chat_validated(message, history):
    """Chat with input validation"""
    # Check length
    if len(message) < 5:
        return " Please ask a longer question (at least 5 characters)."

    if len(message) > 500:
        return " Question too long (max 500 characters)."

    # Check for offensive content (placeholder)
    if is_offensive(message):
        return " Please keep your questions appropriate."

    # Generate response
    return generate_response(message)

demo = gr.ChatInterface(fn=chat_validated)
```

### 3. Caching

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def generate_cached(message):
    """Cache responses for common questions"""
    return generate_response(message)

def chat_cached(message, history):
    return generate_cached(message)

demo = gr.ChatInterface(fn=chat_cached)
```

### 4. Monitoring

```python
import prometheus_client as prom

# Metrics
request_count = prom.Counter('chat_requests_total', 'Total chat requests')
response_time = prom.Histogram('chat_response_seconds', 'Response time')

def chat_monitored(message, history):
    request_count.inc()

    with response_time.time():
        response = generate_response(message)

    return response

demo = gr.ChatInterface(fn=chat_monitored)
```

### 5. Queue Management

```python
demo = gr.ChatInterface(fn=chat)

# Enable queuing
demo.queue(
    concurrency_count=10,  # Max concurrent users
    max_size=100  # Max queue size
)

demo.launch()
```

---

## Troubleshooting

### Model Loading Issues

**Problem**: Model fails to load

**Solutions**:
```python
# 1. Check device
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 2. Use lower precision
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # or torch.int8
    load_in_8bit=True  # for even lower memory
)

# 3. Check authentication
from huggingface_hub import login
login(token="your_token")
```

### Slow Generation

**Solutions**:
```python
# 1. Use vLLM for inference
from vllm import LLM

llm = LLM(model="your-model", gpu_memory_utilization=0.5)

# 2. Reduce max_length
outputs = model.generate(inputs, max_new_tokens=100)

# 3. Use streaming
def generate_stream(message):
    for token in model.generate_stream(message):
        yield token
```

### Interface Not Loading

**Solutions**:
```bash
# 1. Check port
demo.launch(server_port=7860)

# 2. Check firewall
# Allow port 7860

# 3. Use share=True for Colab
demo.launch(share=True)
```

### HF Spaces Issues

**Solutions**:
1. Check logs in Space settings
2. Verify `requirements.txt` includes all dependencies
3. Check hardware requirements (upgrade to GPU if needed)
4. Verify model access (public vs private)
5. Check secrets are set correctly

---

## Complete Example: Production-Ready App

```python
# app.py - Production-ready Gradio app
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
from functools import lru_cache
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model (cached)
@lru_cache(maxsize=1)
def load_model():
    """Load model once and cache"""
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "your-username/grpo-math-model",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("your-username/grpo-math-model")
    logger.info("Model loaded successfully")
    return model, tokenizer

model, tokenizer = load_model()

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

# Rate limiting
last_request_time = {}

def solve_math(message, history, temperature, max_tokens, request: gr.Request):
    """
    Solve math problem with comprehensive error handling and features.
    """
    try:
        # Rate limiting
        client_ip = request.client.host
        now = time.time()

        if client_ip in last_request_time:
            if now - last_request_time[client_ip] < 2:  # 2 seconds
                return "⏱ Please wait a moment before sending another message."

        last_request_time[client_ip] = now

        # Input validation
        if not message or len(message.strip()) < 5:
            return " Please ask a longer question (at least 5 characters)."

        if len(message) > 500:
            return " Question too long. Please keep it under 500 characters."

        # Log request
        logger.info(f"Request from {client_ip}: {message[:50]}...")

        # Generate response
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message}
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(
            inputs,
            max_new_tokens=min(max_tokens, 300),
            temperature=temperature,
            do_sample=True
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract and format
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()

        try:
            reasoning = response.split("<reasoning>")[1].split("</reasoning>")[0].strip()
            answer = response.split("<answer>")[1].split("</answer>")[0].strip()
            formatted = f"** Reasoning:**\n\n{reasoning}\n\n** Answer:** {answer}"
        except:
            formatted = response

        return formatted

    except torch.cuda.OutOfMemoryError:
        logger.error("GPU OOM")
        return " GPU out of memory. Please try a shorter question or lower max_tokens."
    except Exception as e:
        logger.error(f"Error: {e}")
        return " Sorry, something went wrong. Please try again."

# Create interface
demo = gr.ChatInterface(
    fn=solve_math,
    title=" GRPO Math Tutor",
    description="Ask me grade-school math problems! I'll show step-by-step reasoning.",
    examples=[
        "If a pizza is cut into 8 slices and you eat 3, what fraction is left?",
        "A car travels 60 mph for 2.5 hours. How far does it go?",
        "Sarah has $20. She buys 3 books at $4 each. How much money is left?"
    ],
    additional_inputs=[
        gr.Slider(0.1, 1.5, value=0.7, label="Temperature", step=0.1),
        gr.Slider(50, 300, value=200, label="Max Tokens", step=10)
    ],
    theme=gr.themes.Soft(),
    retry_btn=" Retry",
    undo_btn="↩ Undo",
    clear_btn=" Clear"
)

# Enable queue
demo.queue(concurrency_count=10)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
```

---

## Resources

- [Gradio Documentation](https://gradio.app/docs)
- [Gradio Guides](https://gradio.app/guides)
- [HF Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [ChatInterface API](https://gradio.app/docs/chatinterface)
- [Gradio Themes](https://gradio.app/guides/theming-guide)

---

For more guides, see:
- [Prime Intellect Integration](./PRIME_INTELLECT.md)
- [Google Cloud Storage](./GOOGLE_CLOUD_STORAGE.md)
- [Weights & Biases Visualization](./WANDB_VISUALIZATION.md)
