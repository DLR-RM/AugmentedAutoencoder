from setuptools import setup, find_packages

setup(
    name='ae',
    version='0.0.1',
    packages=find_packages(exclude=('tests', 'docs')),
    #install_requires=['pysixd'],
    author='Martin Sundermeyer, Dimitri Henkel',
    author_email='Martin.Sundermeyer@dlr.de, Dimitri.Henkel@dlr.de',
    license='DLR proprietary',
    entry_points={
        'console_scripts': ['ae_init_workspace = ae.ae_init_workspace:main',
                            'ae_train = ae.ae_train:main',
                            'ae_embed = ae.ae_embed:main',
                            'ae_eval = eval.ae_eval:main',
                            'ae_test = test.ae_test:main',
                            'ae_compare = eval.comparative_report:main']
    },
    package_data={'ae': ['cfg/*', 'renderer/shader/*']},
    include_package_data=True
)