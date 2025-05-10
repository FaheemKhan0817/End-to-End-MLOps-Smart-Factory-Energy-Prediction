import os

# Define the project root directory
project_root = "EndToEndMLOpsSmartFactoryEnergyPrediction"

# Define the directory structure with files
structure = {
    "artifacts": {},
    "config": {},
    "custom_jenkins": {},
    "smart_factory_energy_prediction_egg-info": {},  # Matches your project
    "logs": {},
    "mlruns": {},
    "notebooks": {},
    "pipeline": {},
    "src": {},
    "static": {},
    "templates": {},
    "utils": {},
    ".gitignore": "# Git ignore file\n*.pyc\n__pycache__/\n*.log\nmlruns/\nartifacts/\n",
    "app.py": "# FastAPI app for model serving\n",
    "Dockerfile": "# Dockerfile for containerizing the app\n",
    "Jenkinsfile": "// Jenkins pipeline for CI/CD\n",
    "setup.py": """from setuptools import setup, find_packages

# Read requirements from requirements.txt
try:
    with open("requirements.txt", "r") as f:
        requirements = f.read().splitlines()
except FileNotFoundError:
    requirements = []  # Fallback to empty list if requirements.txt is missing

setup(
    name="smart-factory-energy-prediction",
    version="0.1",
    author="Faheem Khan",
    author_email="faheemthakur23@gmail.com",
    description="End to End MLOps Project for Smart Factory Energy Prediction",
    packages=find_packages(),
    install_requires=requirements
)
""",
    "requirements.txt": """pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
statsmodels
pytest
mlflow
docker
"""
}

# Create directories and files
def create_structure(base_path, structure):
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    for name, content in structure.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):  # Directory
            os.makedirs(path, exist_ok=True)
            # Add __init__.py to make it a package
            with open(os.path.join(path, "__init__.py"), "w") as f:
                f.write("# Package initialization file\n")
            create_structure(path, content)  # Recursive call for nested structure
        else:  # File
            with open(path, "w") as f:
                f.write(content)

# Execute the structure creation
create_structure(project_root, structure)

print(f"Project structure created at {project_root}")