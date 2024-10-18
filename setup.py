from setuptools import setup, find_packages

setup(
    name='learnconv', 
    version='0.1.0',  # Initial version
    packages=find_packages('src'),  # Tell setuptools to look in the 'src' folder
    package_dir={'': 'src'},  # Specify the root of packages is 'src'
    install_requires=[
        'numpy',  # required dependencies for the project
    ],
    description='Project for learning convolution operators',
    author='Emilia Magnani',
    author_email='emilia.magnani@uni-tuebingen.de',
    url='https://github.com/EmiliaMagnani/learnconv',  # URL of the project, if applicable
)
