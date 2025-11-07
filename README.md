# MonoCo
10-747 Final Project

## Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the project in development mode
pip install -e .
```

## Augmenting Dataset
Run the following command to run the data augmentation pipeline. All augmented data is automatically stored in augmented_heart_disease.json. Non-augmented data can be pulled in from the UCI API (see uci_data.py).
```bash
python3 data/data_setup.py
```
Uncomment the labelled code in augmentation.py to run real augmentation with ChatGPT.
