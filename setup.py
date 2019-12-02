# -*- coding: utf-8 -*-
from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='cam_l101_ml_for_nlp_e2e_challenge_from_scratch',
    version='1.0',
    packages=['config', 'helpers', 'main', 'model', 'pre_process'],
    url='',
    license='',
    author='ines_blin',
    author_email='ines.blin@student.ecp',
    description='',
    install_requires=requirements,
)