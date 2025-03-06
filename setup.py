from setuptools import setup, find_packages

with open('requirements.txt') as f:
    content = f.readlines()

requirements = [req.strip() for req in content]

setup(
    name='lawyer_bot',
    description='This package contains the code for the lawyer_bot project.',
    packages = find_packages(),
    install_requires=requirements
)
