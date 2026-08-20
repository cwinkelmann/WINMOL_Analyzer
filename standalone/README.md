# WINMOL_Analyzer

Installation in conda env:

    conda create --name WINMOL_Analyzer python==3.9
    conda activate WINMOL_Analyzer

Install tensroflow with GPU on Windows:
    
    conda install -c conda-forge cudatoolkit==11.2.2 cudnn==8.1.0.77
    pip install tensorflow-gpu==2.10.1

Install tensorflow with GPU on Linux / WIndows WSL
    
    pip install tensorflow[and-cuda]

Verify that the GPU is working

    python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

Clone the repo and install the requirements:

    git clone https://github.com/StefanReder/WINMOL_Analyzer
    pip install -r requirements/cpu.txt
    
[optional] Add the conda env as ipykernel to jupyter 
  
    python -m ipykernel install --user --name=WINMOL_Analyzer
    
