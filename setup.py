from setuptools import setup 
import sys 
import os 
import shutil
import distutils.cmd


VERSION = "0.0.1"

class PypiCommand(distutils.cmd.Command):
    
    description = "Build and upload for PyPI."
    user_options = []
    
    def initialize_options(self):
        pass
    
    
    def finalize_options(self):
        pass
    
    
    def run(self):
        try:
            shutil.rmtree("dist/")
        except FileNotFoundError:
            pass
        
        wheel_file = "CytoOne-{}-py3-none-any.whl".format(VERSION)
        tar_file = "CytoOne-{}.tar.gz".format(VERSION)
        
        os.system("{} setup.py sdist bdist_wheel".format(sys.executable))
        os.system("twine upload dist/{} dist/{}".format(wheel_file, tar_file))
    
    
# class CondaCommand(distutils.cmd.Command):
    
#     description = "Build and upload for conda."
#     user_options = []
    
#     def initialize_options(self):
#         pass
    
    
#     def finalize_options(self):
#         pass
    
    
#     def run(self):
#         try:
#             shutil.rmtree("dist_conda/")
#         except FileNotFoundError:
#             pass
#         os.system("conda build . --output-folder dist_conda/")
#         os.system("anaconda upload ./dist_conda/noarch/pMTnet_Omni-{}-py_0.tar.bz2".format(VERSION))


setup(
    name="CytoOne",
    version=VERSION,
    description="A unified probabilistic framework for CyTOF data",
    author="Yuqiu Yang, Kevin Wang, Tao Wang, Xinlei (Sherry) Wang",
    author_email="yuqiuy@smu.edu, kevinwang@mail.smu.edu, Tao.Wang@UTSouthwestern.edu, xinlei.wang@uta.edu",
    long_description_content_type="text/markdown",
    long_description=open("README.md").read(),
    packages=["CytoOne"],
    python_requires=">=3.9,<3.11",
    install_requires=[
        "numpy==1.22.4",
        "pandas==1.5.2",
        "torch==1.13.1",
        "pyro-ppl==1.8.6",
        "matplotlib==3.7.1",
        "seaborn==0.13.0"
    ],
    test_requires=[
        'pytest==7.1.2',
        'coverage==6.3.2',
        'pytest-cov==3.0.0',
        'pytest-mock==3.10.0'
    ],
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
    ],
    cmdclass={
        "pypi": PypiCommand,
        # "conda": CondaCommand
    }
)

