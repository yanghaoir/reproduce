from setuptools import setup, find_packages

setup(
    name='realign',
    version='0.0.1',
    packages=find_packages("src"),
    package_dir={'': 'src'},
    python_requires='>=3.7',
)
