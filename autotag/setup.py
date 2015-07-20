'''
Created on Dec 29, 2014

@author: jordan
'''
from setuptools import setup

setup(name='autotag',
      version='0.456',
      description='Tag suggestion according to text ',
      url='https://github.com/jvalansi/autotag',
      author='Jordan Valansi',
      author_email='jvalansi_autotag@gmail.com',
      license='MIT',
      packages=['autotag'],
      install_requires=[
          'nltk==3.0.0',
      ],
      zip_safe=False)
