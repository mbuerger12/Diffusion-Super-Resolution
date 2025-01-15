# Erde Setup Tutorial with Micromamba and Slurm

## Table of Contents

1. [Connecting to the HPC Cluster](#1-connecting-to-the-hpc-cluster)
2. [Setting Up a Micromamba Environment](#2-setting-up-a-micromamba-environment)
3. [Introduction to Slurm](#3-introduction-to-slurm)
4. [GPU Monitoring and Usage](#4-gpu-monitoring-and-usage)
5. [Important Notes on Storage](#5-important-notes-on-storage)
6. [Helpful Tips](#6-helpful-tips)

---

## 1\. Connecting to the HPC Cluster

Connect to erde with ssh

```
ssh user@erde.rsim.tu-berlin.de
```

Use ssh key authentication (optional)

#### Step 1: Generate SSH Key Pair

Generate ssh key if you don't have one yet.

#### Step 2: Copy the Public Key to the Server

Use the following command to copy your public key to the server:

> ssh-copy-id user@your_server

Replace *"user"* with your server username and *"your_server"* with the server's IP address or domain.

#### Step 3: Verify SSH Key Authentication

Attempt to SSH into the server:

> ssh user@your_server

You should be able to log in without entering a password.

# 2\. Setting Up a Micromamba Environment

Micromamba is a lightweight package manager compatible with conda packages, ideal for managing environments on HPC systems.

### **Install Micromamba**

1. **Download and install micromamba**

   ```
   "${SHELL}" <(curl -L micro.mamba.pm/install.sh)
   ```

   Either set **Prefix Location** in the installation process to */faststorage/$(id -un)/mamba* or lazily overwrite it by running

   ```bash
   echo "export MAMBA_ROOT_PREFIX='/faststorage/$(id -un)/mamba'" >> ~/.bashrc
   ```

   This enforces that all mamba packages are saved to fast storage, (there is more space, and the packages will load much faster)
2. **Reload Shell**:

   ```bash
   source ~/.bashrc  
   ```
3. **Test Installation**:

   ```bash
   micromamba --version
   ```

You should see the version information for micromamba.

### **Create and Activate an Environment**

The pytorch version must be compatible with the CUDA version on the server. By 18.11.24 on erde cuda 12.0 is used. Versions are usually backwards compatible, e.g. 12.1 works with version 12.0.

1. **Create a Micromamba Environment**:

   ```bash
   micromamba create -n pytorch_env python=3.9 -c conda-forge
   ```
2. **Activate the Environment**:

   ```bash
   micromamba activate pytorch_env
   ```
3. **Install PyTorch with CUDA 12.1 Support**:

   ```bash
   micromamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -c conda-forge
   ```

---

## **3\. Introduction to Slurm**

### **Basic Slurm Commands**

* **View available partitions**: `sinfo`
* **Check GPU availability**: `sinfo --format="%P %G %m"`
* **Submit a job**: `sbatch job_script.slurm`
* **Check job status**: `squeue -u $USER`
* **Cancel a job**: `scancel job_id`

### **Example Job Script**

Create a simple Python script to check GPU availability.

**(**`check_gpu.py`)

```python
import torch

def check_gpu():
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device Name:", torch.cuda.get_device_name(0))
        print("CUDA Version:", torch.version.cuda)
    else:
        print("No GPU detected!")

if __name__ == "__main__":
    check_gpu()
```

**(**`job_script.slurm`)

```bash
#!/bin/bash
#SBATCH --job-name=pytorch_job       # Job name
#SBATCH --gres=gpu:1                 # Request one GPU
#SBATCH --cpus-per-task=4            # CPU cores per task
#SBATCH --mem=72G                    # Memory update if necessary
#SBATCH --time=02:00:00              # Time limit
#SBATCH --output=log/pytorch_job_%j.out  # Standard output log (%j is the job ID)
#SBATCH --error=log/pytorch_job_%j.err   # Standard error log

# Run the Python script directly within the micromamba environment
# You can also activate the environment and then run the script
micromamba run -n pytorch_env python check_gpu.py
```

Run it using. If you don't run via SLURM it will not be able to use GPU's as they are managed by SLURM!

```bash
sbatch job_script.slurm
```

---

## **3\. GPU Monitoring and Usage**

### **Using** `nvidia-smi`

* **Monitor GPU Usage**:

  ```bash
  nvidia-smi
  ```
* **Show Detailed Process Info**:

  ```bash
  nvidia-smi -q -d PROCESS
  ```
* **Live Monitoring**:

  ```bash
  watch -n 1 nvidia-smi
  ```

# 5\. Important Notes on Storage

Understanding the storage options on the HPC cluster is crucial for efficient data management.

## Fast Storage vs. Storage Cube

### Fast Storage (`/faststorage`):

* Located on SSDs (Solid State Drives).
* Offers high-speed data access.
* Ideal for storing training data and applications that require fast IO operations.
* Reduces CPU consumption and improves performance.
* There is a wrong path (/media/faststorage) which is empty and should not be used.

### Storage Cube(/media/storagecube):

* Located on HDDs (Hard Disk Drives).
* Slower access speeds compared to SSDs.
* Suitable for archiving data or storing datasets not currently in use.
* Loading data from HDDs can be slow and may cause high IO operations, affecting overall system performance.

### Recommendations:

* **Training Data**: Store on fast storage for quick access and efficient processing.
* **Archived Data**: Store on storage cube if not actively used.
* **Data Management**: Move data between storages as needed. If you require more space, contact your advisor.

## Home Directory

The home directory is on a separate, smaller partition.

### Do Not:

* Save large datasets or unnecessary files in your home directory.
* Store conda or micromamba packages here.

### Do:

* Keep your home directory organized.
* Use it primarily for configuration files and small scripts.

> **Note**: Overloading the home directory can lead to storage issues for you and other users.

---

# 6\. Helpful Tips

## Use Version Control Systems

### Git Repositories:

* Develop your code locally.
* Commit and push changes to the cluster.

## Running Long Python Processes on the Cluster

Tools like `nohup`, `zellij`, or `tmux` keep your processes running even if the SSH connection drops. `zellij`, or `tmux` can be downloaded as binary and moved to the bin path or installed via conda. `tmux`  is also pre-installed on erde,

#### Zellij:

* A beginner-friendly terminal multiplexer.
* Install and use it to manage multiple sessions.

**Example installation:**

```bash
micromamba install -c conda-forge zellij
```

## Remote Development IDEs

Allows you to develop directly on the cluster from your local machine. Also GPU debugging is possible. We strongly recommend development using a debugger.

### PyCharm Professional

* Free for students.

### Visual Studio Code (VSCode)

* Use the Remote SSH extension.

## Monitoring System Resources

### bpytop

An interactive resource monitor for CPU, memory, disk, and network usage.

**Install using micromamba:**

```bash
micromamba install -c conda-forge bpytop
```

**Run it by typing:**

```bash
bpytop
```

## Code Quality Tools

### Pre-Commit Hooks with Ruff

Ruff is a fast Python linter and formatter.

Use pre-commit hooks to automatically check code quality before commits.

Set up a `.pre-commit-config.yaml` file in your repository:

```yaml
  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.0.280
    hooks:
      - id: ruff
```

Install pre-commit in your environment:

```bash
micromamba install -c conda-forge pre-commit
pre-commit install
```