# ROS 2 Interfaces

## Topics

| Topic | Type | Producer | Consumer / purpose |
| --- | --- | --- | --- |
| `/sorting_cell/camera/image_raw` | `sensor_msgs/msg/Image` | Gazebo camera bridge | Default color-perception image input |
| `/sorting_cell/camera/camera_info` | `sensor_msgs/msg/CameraInfo` | Gazebo camera bridge | Intrinsics for image-to-workplane projection |
| `/sorting_cell/detections` | `sorting_cell_interfaces/msg/DetectedObject` | Selected perception source | Pick coordinator and live verifier |
| `/sorting_cell/status` | `sorting_cell_interfaces/msg/CellStatus` | Pick coordinator | Metrics, perception sequencing and live verifier |
| `/sorting_cell/attachment_command` | `std_msgs/msg/String` | Pick coordinator | Gazebo attachment adapter |
| `/sorting_cell/attachment_status` | `std_msgs/msg/String` | Gazebo attachment adapter | Coordinator acknowledgement/failure input |
| `/joint_states` | `sensor_msgs/msg/JointState` | Joint-state broadcaster | Coordinator measured feedback, robot-state publisher and attachment adapter |
| `/clock` | `rosgraph_msgs/msg/Clock` | Gazebo bridge | Simulation time for ROS 2 nodes |

`/sorting_cell/status` uses reliable, transient-local QoS in the coordinator so a verifier that joins shortly after a terminal event can recover it. Camera data uses sensor-data QoS.

## Action

| Action | Type | Client | Server |
| --- | --- | --- | --- |
| `/cell_controller/follow_joint_trajectory` | `control_msgs/action/FollowJointTrajectory` | Pick coordinator | `joint_trajectory_controller` |

The coordinator sends a complete, dense trajectory for one semantic state per goal. A segment advances only after:

1. the action server accepts the goal;
2. the result status is `SUCCEEDED` and the controller error code is `SUCCESSFUL`;
3. a fresh `/joint_states` sample places arm and gripper joints within the configured coordinator tolerance.

Timeout, rejection, controller errors or measured target mismatch produce a terminal `FAILED` status. On an execution timeout, the coordinator requests action cancellation and waits for a terminal result before it can dispatch another object. If termination cannot be confirmed within the default 5 s `cancellation_timeout_s`, or a goal response leaves ownership unknown, it latches a restart-required fault and sends no further goals. The controller topic is not used as a fire-and-forget command interface.

## Service

| Service | Type | Client | Server / purpose |
| --- | --- | --- | --- |
| `/world/sorting_cell/set_pose` | `ros_gz_interfaces/srv/SetEntityPose` | Gazebo attachment adapter | Gazebo bridge; updates the carried model pose |

## `DetectedObject` contract

~~~text
std_msgs/Header header
string object_id
string class_name
geometry_msgs/Pose pose
float32 confidence
~~~

The coordinator requires `header.frame_id == "base_link"`. A physical camera pipeline must perform intrinsic calibration, target depth/workplane estimation and camera-to-base transformation before publishing.

Default scene mappings are:

| Visual class | `object_id` | `class_name` | Route |
| --- | --- | --- | --- |
| Blue | `part_blue` | `accepted` | `accepted_bin` |
| Red | `part_red` | `scratch` | `reject_bin` |
| Orange | `part_orange` | `unknown` | `rework_bin` |

Required workspace validation from `cell.yaml` includes:

- confidence at least `0.70`;
- radial reach from `0.18 m` through `0.69 m`;
- height from `0.055 m` through `0.56 m`;
- absolute lateral coordinate no greater than `0.40 m`;
- valid analytical IK and joint-limit solutions for the generated waypoints.

## `CellStatus` contract

~~~text
std_msgs/Header header
string state
string active_object_id
uint32 completed_cycles
uint32 failed_cycles
float32 current_cycle_time_s
string message
~~~

Nominal state sequence:

~~~text
IDLE -> VALIDATING -> PREGRASP -> APPROACH -> GRASP -> LIFT
     -> TRANSFER -> PLACE -> RELEASE -> RETREAT -> HOME -> COMPLETE
~~~

Invalid input, planning failure, action failure, stale/mismatched measured feedback or attachment failure transitions the active object to `FAILED`. Exactly one `COMPLETE` or `FAILED` event is published for each accepted object ID.

## Attachment command and acknowledgement

The simulator adapter uses a small text protocol on the two attachment topics:

| Direction | Payload | Meaning |
| --- | --- | --- |
| Command | `attach:<object_id>` | Begin carrying the named Gazebo model |
| Command | `detach:<object_id>` | Release the named model |
| Status | `attached:<object_id>:<detail>` | First pose update succeeded; transfer may continue |
| Status | `detached:<object_id>:<detail>` | Release completed; retreat may continue |
| Status | `failed:<object_id>:<detail>` | Adapter rejected, timed out or lost its service operation |

The coordinator checks that acknowledgement object IDs match the active cycle and waits up to its attachment timeout. Real hardware should preserve this success/failure semantic even if its transport is a gripper action, digital I/O or a vendor API.

## Launch arguments

| Argument | Type / values | Default | Effect |
| --- | --- | --- | --- |
| `perception` | `color` or `synthetic` | `color` | Select actual image processing or explicit known-pose test input |
| `gazebo_gui` | boolean | `true` | Start the Gazebo GUI; `false` keeps the simulation server-only |
| `rviz` | boolean | `true` | Start RViz2 |
| `metrics_csv` | path string | empty | Fixed metrics path, or a unique UTC-stamped file under `/tmp` when empty |
| `spawn_delay` | positive seconds | `3.0` | Wait for Gazebo world services before spawning the robot |

Example:

~~~bash
ros2 launch sorting_cell_bringup sorting_cell.launch.py \
  perception:=color gazebo_gui:=false rviz:=false \
  metrics_csv:=/tmp/my_sorting_run.csv
~~~

## Metrics CSV

One row is written for each unique terminal cycle with these columns:

~~~text
timestamp_ns,object_id,state,cycle_time_s,completed,failed,message
~~~

When `metrics_csv` is empty, the generated name is `/tmp/sorting_cell_metrics_<UTC timestamp>.csv`, preventing a normal new run from silently appending to a previous run's default file.

## Runtime evidence

`tools/verify_ros2_demo.py` writes a JSON payload containing each Boolean check plus controller states, camera/joint message counts, detections, terminal counts, queried Gazebo model poses and any failures. The verified 3/3 reference payload is [`evidence/ros2_runtime_verification.json`](evidence/ros2_runtime_verification.json).
