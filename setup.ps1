# Graph-MedSAM environment setup script (PowerShell)

Write-Host "Creating virtual environment..."

python -m venv venv

Write-Host "Activating virtual environment..."

.\venv\Scripts\Activate.ps1

Write-Host "Upgrading pip..."

python -m pip install --upgrade pip

Write-Host "Installing PyTorch with CUDA 12.1..."

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Write-Host "Installing PyTorch Geometric dependencies..."

pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install torch-geometric

Write-Host "Installing SAM / MedSAM dependencies..."

pip install segment-anything

Write-Host "Installing scientific libraries..."

pip install numpy scipy

Write-Host "Installing medical imaging libraries..."

pip install nibabel SimpleITK

Write-Host "Installing data utilities..."

pip install pyyaml tqdm

Write-Host "Installing image processing libraries..."

pip install scikit-image opencv-python

Write-Host "Installing visualization libraries..."

pip install matplotlib seaborn

Write-Host "Installing ML utilities..."

pip install pandas scikit-learn

Write-Host "Installing optional dev tools..."

pip install ipython jupyter

Write-Host "Installation complete."
Write-Host "Run '.\venv\Scripts\Activate.ps1' to activate the environment."
