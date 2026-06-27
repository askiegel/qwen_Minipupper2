from setuptools import setup
from glob import glob
import os

package_name = 'qwen_robot'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')
        ),
        (
            os.path.join('share', package_name, 'config'),
            glob('config/*')
        ),
    ],
    install_requires=[
        'setuptools',
    ],
    zip_safe=True,
    maintainer='Tony Kiegel',
    maintainer_email='askiegel@eagles.usi.edu',
    description='AI-enabled ROS 2 control framework for the Mini Pupper 2 using Qwen',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'qwen_robot = qwen_robot.main:main',
            'qwen_voice = qwen_robot.voice:main',
            'qwen_dashboard = qwen_robot.dashboard:main',
            'vision_check = qwen_robot.vision_check:main',
            'target_check = qwen_robot.target_check:main',
            'backpack_follow = qwen_robot.backpack_follow:main',
        ],
    },
)
