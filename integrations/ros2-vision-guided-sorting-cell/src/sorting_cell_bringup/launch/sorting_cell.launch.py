from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    LogInfo,
    RegisterEventHandler,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _start_after_success(process_name: str, actions: list):
    """Start the next stage only when the preceding one exited successfully."""

    def _on_exit(event, _context):
        if event.returncode == 0:
            return actions
        reason = f"{process_name} failed with exit code {event.returncode}"
        return [LogInfo(msg=f"ERROR: {reason}"), EmitEvent(event=Shutdown(reason=reason))]

    return _on_exit


def _shutdown_on_unexpected_exit(process_name: str):
    """Stop the workcell when critical infrastructure exits unexpectedly."""

    def _on_exit(event, context):
        if context.is_shutdown:
            return []
        reason = f"{process_name} exited unexpectedly with code {event.returncode}"
        return [LogInfo(msg=f"ERROR: {reason}"), EmitEvent(event=Shutdown(reason=reason))]

    return _on_exit


def generate_launch_description() -> LaunchDescription:
    rviz = LaunchConfiguration("rviz")
    perception = LaunchConfiguration("perception")
    gazebo_gui = LaunchConfiguration("gazebo_gui")
    metrics_csv = LaunchConfiguration("metrics_csv")
    spawn_delay = LaunchConfiguration("spawn_delay")
    description_share = Path(get_package_share_directory("sorting_cell_description"))
    gazebo_share = Path(get_package_share_directory("sorting_cell_gazebo"))
    bringup_share = Path(get_package_share_directory("sorting_cell_bringup"))
    ros_gz_share = Path(get_package_share_directory("ros_gz_sim"))
    model_path = description_share / "urdf" / "sorting_arm.urdf.xacro"
    world_path = gazebo_share / "worlds" / "sorting_cell.sdf"
    robot_description = ParameterValue(
        Command([FindExecutable(name="xacro"), " ", str(model_path)]),
        value_type=str,
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(ros_gz_share / "launch" / "gz_sim.launch.py")),
        launch_arguments={
            "gz_args": [
                "-r -v 3 ",
                PythonExpression(
                    [
                        "'' if '",
                        gazebo_gui,
                        "'.lower() in ('true', '1', 'yes', 'on') else '-s '",
                    ]
                ),
                str(world_path),
            ],
            "on_exit_shutdown": "true",
        }.items(),
    )

    state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description, "use_sim_time": True}],
    )

    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-world",
            "sorting_cell",
            "-name",
            "sorting_cell_arm",
            "-topic",
            "robot_description",
            "-x",
            "0",
            "-y",
            "0",
            "-z",
            "0",
        ],
        output="screen",
    )

    clock_and_camera_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/sorting_cell/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image",
            "/sorting_cell/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            "/world/sorting_cell/set_pose@ros_gz_interfaces/srv/SetEntityPose",
        ],
        output="screen",
    )

    joint_state_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
            "--controller-manager-timeout",
            "60",
        ],
        output="screen",
    )
    cell_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "cell_controller",
            "--controller-manager",
            "/controller_manager",
            "--controller-manager-timeout",
            "60",
        ],
        output="screen",
    )

    runtime_nodes = [
        Node(
            package="sorting_cell_core",
            executable="gazebo_attachment",
            output="screen",
            parameters=[{"use_sim_time": True, "world_name": "sorting_cell"}],
        ),
        Node(
            package="sorting_cell_core",
            executable="pick_coordinator",
            output="screen",
            parameters=[{"use_sim_time": True}],
        ),
        Node(
            package="sorting_cell_core",
            executable="cell_metrics",
            output="screen",
            parameters=[
                {
                    "use_sim_time": True,
                    "output_csv": metrics_csv,
                }
            ],
        ),
        Node(
            package="sorting_cell_core",
            executable="color_perception",
            output="screen",
            parameters=[{"use_sim_time": True}],
            condition=IfCondition(PythonExpression(["'", perception, "' == 'color'"])),
        ),
        Node(
            package="sorting_cell_core",
            executable="synthetic_perception",
            output="screen",
            parameters=[{"use_sim_time": True}],
            condition=IfCondition(PythonExpression(["'", perception, "' == 'synthetic'"])),
        ),
    ]
    runtime_exit_handlers = [
        RegisterEventHandler(
            OnProcessExit(
                target_action=node,
                on_exit=_shutdown_on_unexpected_exit(label),
            )
        )
        for label, node in zip(
            (
                "Gazebo attachment node",
                "Pick coordinator",
                "Cell metrics node",
                "Color perception node",
                "Synthetic perception node",
            ),
            runtime_nodes,
        )
    ]

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", str(bringup_share / "rviz" / "sorting_cell.rviz")],
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(rviz),
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("rviz", default_value="true", description="Start RViz2"),
            DeclareLaunchArgument(
                "perception",
                default_value="color",
                description="Perception source: 'color' (camera) or 'synthetic' (explicit test mode)",
                choices=["color", "synthetic"],
            ),
            DeclareLaunchArgument(
                "gazebo_gui",
                default_value="true",
                description="Start the Gazebo GUI; set false for server-only CI runs",
            ),
            DeclareLaunchArgument(
                "metrics_csv",
                default_value="",
                description="Optional metrics CSV path; empty creates a unique /tmp file",
            ),
            DeclareLaunchArgument(
                "spawn_delay",
                default_value="3.0",
                description="Seconds to let Gazebo advertise world services before spawning",
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=state_publisher,
                    on_exit=_shutdown_on_unexpected_exit("Robot state publisher"),
                )
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=clock_and_camera_bridge,
                    on_exit=_shutdown_on_unexpected_exit("Gazebo bridge"),
                )
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=spawn_robot,
                    on_exit=_start_after_success("Robot spawn", [joint_state_spawner]),
                )
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=joint_state_spawner,
                    on_exit=_start_after_success(
                        "Joint-state broadcaster spawner", [cell_controller_spawner]
                    ),
                )
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=cell_controller_spawner,
                    on_exit=_start_after_success("Cell controller spawner", runtime_nodes),
                )
            ),
            *runtime_exit_handlers,
            gazebo,
            state_publisher,
            clock_and_camera_bridge,
            rviz_node,
            TimerAction(period=spawn_delay, actions=[spawn_robot]),
        ]
    )
