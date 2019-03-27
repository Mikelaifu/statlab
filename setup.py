from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='StatLab',
    url='https://github.com/Mikelaifu/statlab',
    author='Mike Wu',
    author_email='mikewu940327@gmail.com',
    # Needed to actually package something
    packages=['statlab'],
    # Needed for dependencies
    install_requires=['numpy', 'pandas', 'matplotlib', 'warnings', 'functools', 'collections', "scipy", "math", "decimal"],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='MIT',
    description='a pacakge to perform statistical analysis, calcultion and visualization'
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)