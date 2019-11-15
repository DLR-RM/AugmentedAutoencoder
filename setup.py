from setuptools import setup, find_packages

setup(
    name='auto_pose',
    version='0.0.1',
    packages=find_packages(exclude=('docs')),
    # install_requires=['imgaug>=0.2.3'],
    install_requires=['progressbar', 'bitarray'],
    author='Martin Sundermeyer, Dimitri Henkel',
    author_email='Martin.Sundermeyer@dlr.de, Dimitri.Henkel@dlr.de',
    license='DLR proprietary',
    entry_points={
        'console_scripts': ['ae_init_workspace = auto_pose.ae.ae_init_workspace:main',
                            'ae_train = auto_pose.ae.ae_train:main',
                            'ae_embed = auto_pose.ae.ae_embed:main',
                            'ae_embed_multi = auto_pose.ae.ae_embed_multi:main',
                            'ae_eval = auto_pose.eval.ae_eval:main',
                            'ae_test_embedding = auto_pose.test.ae_test_embedding:main',
                            'ae_compare = auto_pose.eval.comparative_report:main']
    },
    package_data={'auto_pose': ['ae/cfg/*', 'ae/cfg_eval/*', 'ae/cfg_m3vision/*', 'm3_interface/sample_data/*', 'meshrenderer/shader/*']},
    # include_package_data=True
)