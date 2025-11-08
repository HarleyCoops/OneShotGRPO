# Google Cloud Storage Integration for Colab

##  Table of Contents
- [Overview](#overview)
- [Why Use Google Cloud Storage?](#why-use-google-cloud-storage)
- [Setup](#setup)
- [Quick Start](#quick-start)
- [Checkpoint Management](#checkpoint-management)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Cost Optimization](#cost-optimization)
- [Alternatives](#alternatives)

---

## Overview

This guide shows you how to save model checkpoints to Google Cloud Storage (GCS) from Google Colab, ensuring your training progress persists even if your Colab session disconnects.

### Key Benefits

- ** Persistence**: Survives Colab disconnects and restarts
- ** Unlimited Storage**: No 15GB Drive limit
- ** Fast Transfers**: Better upload/download speeds than Drive
- ** Versioning**: Built-in checkpoint history
- ** Team Sharing**: Easy collaboration with team members
- ** Global Access**: Access from anywhere

---

## Why Use Google Cloud Storage?

### Comparison with Alternatives

| Feature | GCS | Google Drive | Local Colab |
|---------|-----|--------------|-------------|
| **Persistence** |  Permanent |  Permanent |  Session only |
| **Size Limit** | Unlimited* | 15GB free | ~100GB |
| **Upload Speed** | Fast | Moderate | N/A |
| **Download Speed** | Fast | Moderate | N/A |
| **Team Sharing** |  Easy |  Limited |  No |
| **Versioning** |  Built-in |  Manual |  No |
| **Cost** | $0.02/GB/month | Free (15GB) | Free |
| **API Access** |  Excellent |  Limited |  No |

\* *Subject to billing limits*

### When to Use GCS

**Choose GCS if you:**
- Train large models (>1B parameters)
- Need guaranteed persistence
- Work in teams
- Want fast checkpoint access
- Train for >6 hours
- Need version control

**Choose Google Drive if you:**
- Train small models (<500M parameters)
- Work solo
- Stay under 15GB
- Don't need fast access

---

## Setup

### Prerequisites

1. **Google Cloud Account**
   - Create at [cloud.google.com](https://cloud.google.com)
   - $300 free credit for new users

2. **Google Cloud Project**
   - Create a project in [Console](https://console.cloud.google.com)
   - Enable billing (required even with free credits)

3. **GCS Bucket**
   - Create a bucket for your checkpoints
   - Choose appropriate region

### Step 1: Create GCS Bucket

#### Via Web Console

1. Go to [console.cloud.google.com/storage](https://console.cloud.google.com/storage)
2. Click "Create Bucket"
3. Configure:
   ```
   Name: your-grpo-checkpoints
   Location: us-central1 (or nearest to Colab)
   Storage class: Standard
   Access control: Uniform
   ```
4. Click "Create"

#### Via Command Line

```bash
# Set variables
PROJECT_ID="your-project-id"
BUCKET_NAME="your-grpo-checkpoints"
REGION="us-central1"

# Create bucket
gcloud storage buckets create gs://${BUCKET_NAME} \
    --project=${PROJECT_ID} \
    --location=${REGION} \
    --uniform-bucket-level-access
```

### Step 2: Set Up Authentication

#### In Google Colab

```python
from google.colab import auth
from google.cloud import storage

# Authenticate
auth.authenticate_user()

# Set project
PROJECT_ID = "your-project-id"  # Replace with your project
!gcloud config set project {PROJECT_ID}

# Test connection
client = storage.Client(project=PROJECT_ID)
buckets = list(client.list_buckets())
print(f" Connected! Found {len(buckets)} buckets")
```

### Step 3: Install Required Libraries

```python
# Usually pre-installed in Colab, but just in case:
!pip install -q google-cloud-storage
```

---

## Quick Start

### Basic Checkpoint Saving

```python
from google.cloud import storage
import os

# Configuration
PROJECT_ID = "your-project-id"
BUCKET_NAME = "your-grpo-checkpoints"
MODEL_NAME = "qwen-grpo-v1"

# Initialize client
storage_client = storage.Client(project=PROJECT_ID)
bucket = storage_client.bucket(BUCKET_NAME)

# Save a checkpoint
def save_checkpoint_to_gcs(local_path, gcs_path):
    """Upload a checkpoint directory to GCS"""
    for root, dirs, files in os.walk(local_path):
        for file in files:
            local_file = os.path.join(root, file)
            relative_path = os.path.relpath(local_file, local_path)
            blob_path = os.path.join(gcs_path, relative_path)

            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_file)
            print(f" Uploaded: {blob_path}")

# Example usage
save_checkpoint_to_gcs(
    local_path="/content/outputs/checkpoint-1000",
    gcs_path=f"{MODEL_NAME}/checkpoint-1000"
)
```

### Loading from GCS

```python
def load_checkpoint_from_gcs(gcs_path, local_path):
    """Download a checkpoint directory from GCS"""
    os.makedirs(local_path, exist_ok=True)

    blobs = bucket.list_blobs(prefix=gcs_path)
    for blob in blobs:
        # Skip directories
        if blob.name.endswith('/'):
            continue

        # Compute local file path
        relative_path = os.path.relpath(blob.name, gcs_path)
        local_file = os.path.join(local_path, relative_path)

        # Create parent directories
        os.makedirs(os.path.dirname(local_file), exist_ok=True)

        # Download
        blob.download_to_filename(local_file)
        print(f" Downloaded: {blob.name}")

# Example usage
load_checkpoint_from_gcs(
    gcs_path=f"{MODEL_NAME}/checkpoint-1000",
    local_path="/content/outputs/checkpoint-1000"
)
```

---

## Checkpoint Management

### Auto-Save During Training

#### Option 1: Save After Each Checkpoint

```python
from transformers import TrainerCallback

class GCSCheckpointCallback(TrainerCallback):
    """Automatically upload checkpoints to GCS"""

    def __init__(self, bucket, gcs_prefix):
        self.bucket = bucket
        self.gcs_prefix = gcs_prefix

    def on_save(self, args, state, control, **kwargs):
        # Get the latest checkpoint directory
        checkpoint_dir = f"{args.output_dir}/checkpoint-{state.global_step}"

        if os.path.exists(checkpoint_dir):
            print(f" Uploading checkpoint-{state.global_step} to GCS...")
            gcs_path = f"{self.gcs_prefix}/checkpoint-{state.global_step}"
            save_checkpoint_to_gcs(checkpoint_dir, gcs_path)
            print(f" Uploaded to gs://{self.bucket.name}/{gcs_path}")

# Add to trainer
trainer = GRPOTrainer(
    # ... other args
    callbacks=[GCSCheckpointCallback(bucket, MODEL_NAME)]
)
```

#### Option 2: Background Upload

```python
import threading

class AsyncGCSUploader:
    """Upload checkpoints to GCS in background"""

    def __init__(self, bucket, gcs_prefix):
        self.bucket = bucket
        self.gcs_prefix = gcs_prefix
        self.upload_queue = []
        self.thread = None

    def upload_async(self, local_path, step):
        """Queue checkpoint for upload"""
        self.upload_queue.append((local_path, step))
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self._process_queue)
            self.thread.start()

    def _process_queue(self):
        """Process upload queue"""
        while self.upload_queue:
            local_path, step = self.upload_queue.pop(0)
            gcs_path = f"{self.gcs_prefix}/checkpoint-{step}"
            save_checkpoint_to_gcs(local_path, gcs_path)

# Usage
uploader = AsyncGCSUploader(bucket, MODEL_NAME)
uploader.upload_async("/content/outputs/checkpoint-1000", 1000)
```

### List Available Checkpoints

```python
def list_checkpoints(gcs_prefix):
    """List all checkpoints in GCS"""
    blobs = bucket.list_blobs(prefix=gcs_prefix)
    checkpoints = set()

    for blob in blobs:
        # Extract checkpoint number
        if "checkpoint-" in blob.name:
            parts = blob.name.split("checkpoint-")
            if len(parts) > 1:
                checkpoint_num = parts[1].split("/")[0]
                if checkpoint_num.isdigit():
                    checkpoints.add(int(checkpoint_num))

    return sorted(checkpoints)

# Example
checkpoints = list_checkpoints(MODEL_NAME)
print(f"Available checkpoints: {checkpoints}")
```

### Resume from Latest Checkpoint

```python
def get_latest_checkpoint(gcs_prefix):
    """Get the latest checkpoint number"""
    checkpoints = list_checkpoints(gcs_prefix)
    return max(checkpoints) if checkpoints else None

# Resume training
latest = get_latest_checkpoint(MODEL_NAME)
if latest:
    print(f"Found checkpoint-{latest}, downloading...")
    load_checkpoint_from_gcs(
        gcs_path=f"{MODEL_NAME}/checkpoint-{latest}",
        local_path="/content/outputs/checkpoint-latest"
    )
    resume_from_checkpoint = "/content/outputs/checkpoint-latest"
else:
    print("No checkpoints found, starting fresh")
    resume_from_checkpoint = None
```

### Delete Old Checkpoints

```python
def cleanup_old_checkpoints(gcs_prefix, keep_last_n=5):
    """Keep only the last N checkpoints"""
    checkpoints = list_checkpoints(gcs_prefix)

    if len(checkpoints) <= keep_last_n:
        print(f"Only {len(checkpoints)} checkpoints, nothing to delete")
        return

    to_delete = checkpoints[:-keep_last_n]
    print(f"Deleting {len(to_delete)} old checkpoints: {to_delete}")

    for checkpoint_num in to_delete:
        prefix = f"{gcs_prefix}/checkpoint-{checkpoint_num}/"
        blobs = bucket.list_blobs(prefix=prefix)

        for blob in blobs:
            blob.delete()
            print(f" Deleted: {blob.name}")

# Example
cleanup_old_checkpoints(MODEL_NAME, keep_last_n=3)
```

---

## Best Practices

### 1. Compression

Compress checkpoints before upload to save time and money:

```python
import shutil

def compress_and_upload(local_path, gcs_path):
    """Compress checkpoint and upload to GCS"""
    # Create tar.gz archive
    archive_name = f"{local_path}.tar.gz"
    shutil.make_archive(
        local_path,
        'gztar',
        root_dir=os.path.dirname(local_path),
        base_dir=os.path.basename(local_path)
    )

    # Upload compressed file
    blob = bucket.blob(f"{gcs_path}.tar.gz")
    blob.upload_from_filename(archive_name)

    # Cleanup
    os.remove(archive_name)
    print(f" Uploaded compressed: {gcs_path}.tar.gz")

def download_and_decompress(gcs_path, local_path):
    """Download and decompress checkpoint"""
    # Download
    archive_name = f"{local_path}.tar.gz"
    blob = bucket.blob(f"{gcs_path}.tar.gz")
    blob.download_to_filename(archive_name)

    # Decompress
    shutil.unpack_archive(archive_name, os.path.dirname(local_path))

    # Cleanup
    os.remove(archive_name)
    print(f" Downloaded and decompressed: {local_path}")
```

### 2. Metadata Tracking

Add metadata to track checkpoint info:

```python
def save_with_metadata(local_path, gcs_path, metadata):
    """Save checkpoint with metadata"""
    # Upload files
    save_checkpoint_to_gcs(local_path, gcs_path)

    # Save metadata
    metadata_blob = bucket.blob(f"{gcs_path}/metadata.json")
    metadata_blob.upload_from_string(
        json.dumps(metadata, indent=2),
        content_type="application/json"
    )

# Example usage
save_with_metadata(
    local_path="/content/outputs/checkpoint-1000",
    gcs_path=f"{MODEL_NAME}/checkpoint-1000",
    metadata={
        "step": 1000,
        "loss": 0.234,
        "timestamp": datetime.now().isoformat(),
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "dataset_size": 7500,
        "hyperparameters": {...}
    }
)
```

### 3. Progress Tracking

Show upload/download progress:

```python
from tqdm import tqdm

def save_with_progress(local_path, gcs_path):
    """Upload with progress bar"""
    # Count files
    file_list = []
    for root, dirs, files in os.walk(local_path):
        for file in files:
            file_list.append(os.path.join(root, file))

    # Upload with progress
    with tqdm(total=len(file_list), desc="Uploading") as pbar:
        for local_file in file_list:
            relative_path = os.path.relpath(local_file, local_path)
            blob_path = os.path.join(gcs_path, relative_path)

            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_file)

            pbar.update(1)
            pbar.set_postfix_str(f"Current: {relative_path}")
```

### 4. Error Handling

Robust error handling for network issues:

```python
from google.api_core import retry

@retry.Retry(predicate=retry.if_exception_type(Exception))
def upload_with_retry(local_file, blob_path):
    """Upload with automatic retry on failure"""
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_file)
    return blob

def safe_upload(local_path, gcs_path, max_retries=3):
    """Upload with error handling"""
    try:
        save_checkpoint_to_gcs(local_path, gcs_path)
        return True
    except Exception as e:
        print(f" Upload failed: {e}")
        return False
```

---

## Troubleshooting

### Permission Denied

**Error**: `403 Forbidden` or `Permission Denied`

**Solutions**:
1. Verify authentication: `auth.authenticate_user()`
2. Check project access: `gcloud auth list`
3. Verify bucket permissions in Console
4. Ensure billing is enabled

### Slow Uploads

**Solutions**:
1. Compress checkpoints before upload
2. Use multi-threaded uploads
3. Choose bucket in same region as Colab
4. Upload only essential files (exclude optimizer states)

### Quota Exceeded

**Error**: `429 Too Many Requests`

**Solutions**:
1. Implement rate limiting
2. Use exponential backoff
3. Request quota increase in Console
4. Spread uploads over time

### Out of Space

**Error**: `Insufficient storage`

**Solutions**:
1. Delete old checkpoints: `cleanup_old_checkpoints()`
2. Enable lifecycle management in bucket
3. Use cheaper storage class for old checkpoints

---

## Cost Optimization

### Storage Classes

Choose appropriate storage class based on access patterns:

| Class | Cost/GB/Month | Best For |
|-------|---------------|----------|
| Standard | $0.020 | Frequent access (<1 month) |
| Nearline | $0.010 | Monthly access |
| Coldline | $0.004 | Quarterly access |
| Archive | $0.0012 | Yearly access |

### Cost-Saving Tips

1. **Use Lifecycle Rules**
   ```python
   # Auto-delete checkpoints after 30 days
   bucket.lifecycle_rules = [{
       'action': {'type': 'Delete'},
       'condition': {
           'age': 30,
           'matchesPrefix': [f'{MODEL_NAME}/checkpoint-']
       }
   }]
   bucket.patch()
   ```

2. **Compress Checkpoints**
   - Saves ~50% storage
   - Reduces transfer costs
   - Faster uploads/downloads

3. **Selective Uploads**
   ```python
   # Only upload model weights, not optimizer
   EXCLUDE_PATTERNS = ['optimizer.pt', 'scheduler.pt', 'rng_state.pth']

   def should_upload(filename):
       return not any(pattern in filename for pattern in EXCLUDE_PATTERNS)
   ```

4. **Regional Buckets**
   - Use bucket in same region as Colab
   - Reduces transfer costs
   - Faster uploads

### Cost Monitoring

```python
def estimate_storage_cost(gcs_prefix):
    """Estimate monthly storage cost"""
    blobs = bucket.list_blobs(prefix=gcs_prefix)
    total_size_gb = sum(blob.size for blob in blobs) / (1024**3)

    cost_per_gb = 0.020  # Standard storage
    monthly_cost = total_size_gb * cost_per_gb

    print(f"Total Size: {total_size_gb:.2f} GB")
    print(f"Estimated Monthly Cost: ${monthly_cost:.2f}")

    return monthly_cost
```

---

## Alternatives

### Google Drive Integration

If GCS is overkill, use Google Drive:

```python
from google.colab import drive

# Mount Drive
drive.mount('/content/drive')

# Save checkpoint
import shutil
shutil.copytree(
    '/content/outputs/checkpoint-1000',
    '/content/drive/MyDrive/grpo_checkpoints/checkpoint-1000'
)
```

**Pros**: Free (15GB), simple
**Cons**: Slow, limited storage, no versioning

### Hybrid Approach

Use both GCS and Drive:

```python
# Save to both
save_checkpoint_to_gcs(local_path, gcs_path)  # Primary backup
shutil.copytree(local_path, drive_path)        # Quick access
```

---

## Integration with GRPO Training

### Complete Example

```python
from google.colab import auth
from google.cloud import storage
from transformers import GRPOTrainer, GRPOConfig

# Setup GCS
auth.authenticate_user()
storage_client = storage.Client(project="your-project")
bucket = storage_client.bucket("your-grpo-checkpoints")

# Training config
MODEL_NAME = "qwen-grpo-v1"
training_args = GRPOConfig(
    output_dir="/content/outputs",
    save_steps=100,
    save_total_limit=3,  # Keep last 3 locally
    # ... other args
)

# Create callback
class GCSCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = f"{args.output_dir}/checkpoint-{state.global_step}"
        gcs_path = f"{MODEL_NAME}/checkpoint-{state.global_step}"
        save_checkpoint_to_gcs(checkpoint_dir, gcs_path)

# Train
trainer = GRPOTrainer(
    # ... args
    callbacks=[GCSCallback()]
)
trainer.train()

# After training, keep only best checkpoints
cleanup_old_checkpoints(MODEL_NAME, keep_last_n=5)
```

---

## Next Steps

1.  Set up GCS bucket
2.  Test upload/download
3.  Integrate with training
4.  Set up lifecycle rules
5.  Monitor costs

For more guides, see:
- [Prime Intellect Integration](./PRIME_INTELLECT.md)
- [Weights & Biases Setup](./WANDB_VISUALIZATION.md)
- [Gradio Deployment](./GRADIO_DEPLOYMENT.md)
