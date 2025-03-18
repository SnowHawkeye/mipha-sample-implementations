# Sample Implementations of the MIPHA Framework

## Introduction

This document specifies the expected deliverables for the G1-G2 MIPHA project @CentraleLille.

The repository contains:

- Data samples to test the MIPHA platform: [`data/`](data)
- Sample implementations of the MIPHA framework: [`implementations/`](implementations)
- A jupyter notebook showing how to load the data and use the implementations: [
  `getting_started.ipynb`](getting_started.ipynb)
- An overview on the MIPHA framework and the associated platform: [`mipha_overview.md`](mipha_overview.md)
- The expected specifications of the MIPHA platform: [`specifications.md`](specifications.md)
-

## How to use

### Step 1 - Create a Virtual Environment

To keep dependencies isolated, it's recommended to use a virtual environment. If you're more comfortable / familiar with
another solution to create environments (such as conda), feel free to skip this step!

```sh
python -m venv venv
```

On macOS/Linux, activate it:

```sh
source venv/bin/activate
```

On Windows:

```sh
venv\Scripts\activate
```

### Step 2 - Install Dependencies

Once the virtual environment is active, install the required dependencies:

```sh
pip install -r requirements.txt
```

### Step 3 - Run the notebook

The [`getting_started.ipynb`](getting_started.ipynb) notebook should now be able to run with all the required libraries.
Head over there to see how to unpack the datasets and use the MIPHA implementations!


