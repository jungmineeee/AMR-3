from setuptools import find_packages, setup
import glob
import os

package_name = 'camera_publish'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob.glob(os.path.join('launch', '*.launch.py'))),
        ('share/' + package_name + '/param', glob.glob(os.path.join('param', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Hyunwoo-kim',
    maintainer_email='aq3480@gmail.com',
    description='Camera publisher and Main GUI',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_publish = camera_publish.camera_publish:main',
        ],
    },
)