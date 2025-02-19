
#!/bin/bash

# Step 1: Create and activate a virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created."
fi

source venv/bin/activate  # Use "venv\Scripts\activate" for Windows

# Step 2: Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Step 3: Run the preprocessing scripts
python src/preprocess.py  # Update path if needed

echo "Setup completed successfully!"

