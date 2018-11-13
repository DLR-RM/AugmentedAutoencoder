from setuptools import setup, find_packages

setup(
    name='auto_pose',
    version='0.9',
    packages=find_packages(exclude=('docs')),
    install_requires=[],
    author='Martin Sundermeyer',
    author_email='Martin.Sundermeyer@dlr.de',
    license='MIT license',
    entry_points={
        'console_scripts': ['ae_init_workspace = auto_pose.ae.ae_init_workspace:main',
                            'ae_train = auto_pose.ae.ae_train:main',
                            'ae_embed = auto_pose.ae.ae_embed:main',
                            'ae_eval = auto_pose.eval.ae_eval:main']
    },
    package_data={'auto_pose': ['ae/cfg/*', 'ae/cfg_eval/*', 'ae/cfg_m3vision/*', 'm3_interface/sample_data/*', 'meshrenderer/shader/*']},
    # include_package_data=True
)