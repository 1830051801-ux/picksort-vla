from glob import glob
from setuptools import find_packages, setup

package_name = "sorting_cell_core"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "numpy", "PyYAML"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="Zhong Jiawei",
    maintainer_email="1830051801@qq.com",
    description="Core planning, safety and ROS 2 nodes for the robot sorting workcell.",
    license="MIT",
    url="https://github.com/1830051801-ux/ros2-vision-guided-sorting-cell",
    project_urls={
        "Source": "https://github.com/1830051801-ux/ros2-vision-guided-sorting-cell",
        "Issues": "https://github.com/1830051801-ux/ros2-vision-guided-sorting-cell/issues",
    },
    entry_points={
        "console_scripts": [
            "color_perception = sorting_cell_core.nodes.color_perception_node:main",
            "synthetic_perception = sorting_cell_core.nodes.synthetic_perception_node:main",
            "pick_coordinator = sorting_cell_core.nodes.pick_coordinator_node:main",
            "gazebo_attachment = sorting_cell_core.nodes.gazebo_attachment_node:main",
            "cell_metrics = sorting_cell_core.nodes.metrics_node:main",
        ],
    },
)
