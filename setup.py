import setuptools

__version__ = "1.0rc1"
__author__ = "Ricardo Montañana Gómez"

def readme():
    with open('README.md') as f:
        return f.read()


setuptools.setup(
    name='N_Network',
    version=__version__,
    license='MIT License',
    description='A personal implementation of a Neural Network',
    long_description=readme(),
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    url='https://github.com/doctorado-ml/neuralnetwork',
    author=__author__,
    author_email='ricardo.montanana@alu.uclm.es',
    keywords='neural_network',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research'
    ],
    install_requires=[
        'scikit-learn>=0.23.0',
        'numpy',
        'matplotlib',
        'seaborn'
    ],
    zip_safe=False
)
