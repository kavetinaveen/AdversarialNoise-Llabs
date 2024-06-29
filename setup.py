from setuptools import setup, find_packages

VERSION = '0.0.5' 
DESCRIPTION = 'Adversarial Noise'
LONG_DESCRIPTION = 'Introduce adversarial noise into an input image to trick an image classification model into misclassifying it as the desired target class.'

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="AdNoise", 
    version=VERSION,
    author="Naveen Kaveti",
    author_email="kaveti.naveenkumar@gmail.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[], 
    
    keywords=['python', 'adversarial noise'],
    classifiers= [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)