# Initialize the virtual environment (CPU version)
init-cpu:
	@echo "Initializing CPU environment..."
	python3 -m venv venv
	@echo "Activating environment and installing dependencies..."
	./venv/bin/python -m pip install --upgrade pip
	./venv/bin/python -m pip install -r requirements.txt
	./venv/bin/python -m pip install pandas

# Install DVC
init-dvc:
	@echo "Initializing DVC environment..."
	./venv/bin/python -m pip install dvc
	./venv/bin/python -m pip install dvc[gdrive]
	./venv/bin/dvc init
	./venv/bin/dvc remote add -d myremote gdrive://1p1SZiOeMQuPIRZnoTQM9YBORmsp5CjM9
	./venv/bin/git add .dvc/config
	./venv/bin/git commit -m "Setup DVC with Google Drive"
	./venv/bin/dvc push

# Run preprocessing
preprocess:
	@echo "Running preprocessing pipeline..."
	./venv/bin/python src/preprocess.py

# Run training pipeline
train:
	@echo "Running ML training pipeline..."
	./venv/bin/python src/train.py

# Run model evaluation
evaluate:
	@echo "Evaluating trained models..."
	./venv/bin/python src/evaluate.py

# Run prediction pipeline
predict:
	@echo "Making predictions with the best model..."
	./venv/bin/python src/predict.py

# Run MLflow UI
mlflow-ui:
	@echo "Launching MLflow UI..."
	./venv/bin/mlflow ui --host 0.0.0.0 --port 5000

# Run MLflow server
mlflow-server:
    @echo "Launching MLflow server..."
    ./venv/bin/mlflow server --host 127.0.0.1 --port 5000