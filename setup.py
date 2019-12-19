
from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='canepwdl',
    version='0.1.0',
    description='Next event prediction with deep learning',
    long_description=readme,
    author='Sven Weinzierl',
    author_email='sven.weinzierl@fau.de',
    url='https://github.com/fau-is/next-event-predicton-with-deep-learning',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

