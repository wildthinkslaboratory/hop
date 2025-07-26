from setuptools import find_packages, setup

package_name = 'hop'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/launch.py']),
    ],
    install_requires=['setuptools', 'numpy', 'casadi'],
    zip_safe=True,
    maintainer='izzy',
    maintainer_email='izzymones@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'nmpc_controller = hop.nmpc_controller:main'
            # 'test = hop.test:main',
        ],
    },
)
