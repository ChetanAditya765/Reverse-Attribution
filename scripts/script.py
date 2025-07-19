import os
import shutil

# Create the complete project structure
project_structure = {
    'reverse-attribution': [
        'core/__init__.py',
        'core/ra.py',
        'core/explainer_utils.py', 
        'core/model_utils.py',
        'core/visualizer.py',
        'datasets/__init__.py',
        'datasets/dataset_utils.py',
        'datasets/download_datasets.py',
        'models/__init__.py',
        'models/model_factory.py',
        'models/bert_sentiment.py',
        'models/resnet_cifar.py',
        'evaluation/__init__.py',
        'evaluation/evaluate.py',
        'evaluation/metrics.py',
        'evaluation/user_study.py',
        'tests/__init__.py',
        'tests/test_ra_core.py',
        'tests/test_evaluation.py',
        'tests/test_models.py',
        'examples/__init__.py',
        'examples/reproduce_results.py',
        'examples/custom_model_example.py',
        'scripts/setup_environment.py',
        'scripts/download_models.py',
        'docs/api_reference.md',
        'docs/user_guide.md',
        'app.py',
        'requirements.txt',
        'environment.yml',
        'setup.py',
        'README.md',
        'LICENSE',
        '.gitignore'
    ]
}

# Create directory structure
base_dir = 'reverse-attribution'
if os.path.exists(base_dir):
    shutil.rmtree(base_dir)

for path in project_structure[base_dir]:
    full_path = os.path.join(base_dir, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
    # Create empty files
    with open(full_path, 'w') as f:
        f.write('')

print("Project structure created successfully!")
print(f"Total files created: {len(project_structure[base_dir])}")

# List the created structure
for root, dirs, files in os.walk(base_dir):
    level = root.replace(base_dir, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    sub_indent = ' ' * 2 * (level + 1)
    for file in files:
        print(f'{sub_indent}{file}')