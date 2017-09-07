from setuptools import setup, find_packages

setup(
    name='ae',
    version='0.0.1',
    packages=find_packages(exclude=('tests', 'docs')),
    #install_requires=['setuptools', 'progressbar'],
    author='Dimitri Henkel, Martin Sundermeyer',
    author_email='Dimitri.Henkel@dlr.de, Martin.Sundermeyer@dlr.de',
    license='DLR proprietary',
    entry_points={
        'console_scripts': ['ae_init_workspace = ae.ae_init_workspace:main',
                            'ae_train = ae.ae_train:main',
                            'ae_embed = ae.ae_embed:main']
    },
    package_data={'ae': ['cfg/*', 'renderer/shader/*']},
    include_package_data=True
)