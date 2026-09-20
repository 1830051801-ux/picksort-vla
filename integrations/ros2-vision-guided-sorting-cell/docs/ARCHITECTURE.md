# Vision-Guided Sorting Workcell Architecture

The project separates deterministic cell logic from ROS 2 transport and simulator adapters. The planning core can be tested on Windows without ROS, while the complete runtime closes the camera-to-part-motion loop in ROS 2 Jazzy and Gazebo Harmonic.

## End-to-end data and feedback flow

~~~mermaid
flowchart LR
    CAM["Gazebo RGB camera<br/>Image + CameraInfo"] --> VISION["Color perception<br/>HSV + ROI + stability"]
    VISION --> DET["DetectedObject<br/>frame: base_link"]
    DET --> SAFE["Workspace and confidence validation"]
    SAFE --> PLAN["Analytical IK + quintic trajectories"]
    PLAN --> COORD["Pick coordinator<br/>segment state machine"]
    COORD -->|"FollowJointTrajectory goals"| CTRL["ros2_control<br/>cell_controller"]
    CTRL --> SIM["Gazebo arm + gripper"]
    SIM -->|"measured JointState"| COORD
    COORD -->|"attach/detach command"| ATTACH["Gazebo attachment adapter"]
    ATTACH -->|"attached/detached/failed ack"| COORD
    ATTACH -->|"SetEntityPose"| PART["Gazebo workpiece"]
    COORD --> STATUS["CellStatus"]
    STATUS --> METRICS["Timestamped CSV metrics"]
~~~

No segment is treated as complete merely because points were published. The coordinator requires a successful controller action result and fresh measured joint positions within tolerance. Likewise, the grasp/release state machine waits for attachment acknowledgement before it continues.

## Package boundaries

| Package | Responsibility |
| --- | --- |
| `sorting_cell_interfaces` | Stable `DetectedObject` and `CellStatus` contracts |
| `sorting_cell_description` | Robot geometry, joints, limits, inertials and `gz_ros2_control` interfaces |
| `sorting_cell_gazebo` | Workcell world, camera, parts, collision trays and controller parameters |
| `sorting_cell_core` | Image perception, IK, safety, trajectory generation, workflow, coordinator, attachment and metrics nodes |
| `sorting_cell_bringup` | Ordered startup of Gazebo, robot, controllers, runtime nodes and RViz |

## Ordered startup

The launch file starts dependencies in this order:

1. Gazebo, robot-state publisher, bridges and optional RViz start.
2. The robot is spawned into Gazebo.
3. A successful spawn starts `joint_state_broadcaster`.
4. A successful broadcaster start activates `cell_controller`.
5. Only after both controllers are ready do perception, coordinator, attachment and metrics nodes start.

A failed spawn or controller-spawner process shuts down the launch instead of allowing a partially initialized cell to continue.

## Perception boundary

`perception:=color` is the default. `color_perception` consumes the actual Gazebo camera's `sensor_msgs/Image` and `CameraInfo`, then performs:

- BGR/HSV image decoding and blue, red and orange segmentation;
- pickup-region, contour area and apparent-size filtering;
- projection onto the known workpiece plane in `base_link`;
- multi-frame position stability gating;
- ordered publication and retry until a terminal `CellStatus` is observed.

The simulator's fixed camera geometry makes the direct plane projection deterministic. A physical installation must replace that geometry assumption with calibrated intrinsics, hand-eye calibration and TF2 transformation.

`perception:=synthetic` bypasses camera processing and publishes known scene poses. It is retained only for explicit transport/planner tests; it is not the default or the source used for the committed ROS runtime evidence.

## Planning and action execution

`config/cell.yaml` is the authoritative routing and workspace configuration. `WorkflowPlanner.from_yaml()` loads confidence limits, workspace bounds, gripper positions, destinations and class-to-bin routes.

For each accepted detection, the coordinator:

1. waits for the `FollowJointTrajectory` action server and fresh states for all six controlled joints;
2. uses measured `/joint_states`, not a presumed home pose, as the plan start;
3. creates dense multi-point semantic segments for pregrasp, approach, grasp, lift, transfer, place, release, retreat and home;
4. sends one action goal per segment and checks acceptance, timeout, action status and controller result;
5. verifies fresh measured arm and gripper error against the segment target;
6. publishes exactly one terminal `COMPLETE` or `FAILED` event per object.

The controller uses its own goal constraints. Coordinator-side measurement verification is an additional application-level guard and remains required even after the action reports success.

An execution timeout does not release the controller for the next object immediately. The coordinator requests cancellation and waits for the action to reach a terminal state. If cancellation is rejected, its response fails, no terminal result arrives within `cancellation_timeout_s`, or goal ownership/terminal state is otherwise unknown, the coordinator publishes one failure and latches a fault that blocks all further goals until the node is restarted.

## Attachment adapter

Gazebo has no physical finger-contact grasp model in this project, so `gazebo_attachment` is the simulator-specific grasp adapter. While attached, it computes the tool pose from measured joints and calls `/world/sorting_cell/set_pose` for the named model.

The adapter publishes `attached:<id>:<detail>` only after the first successful pose-service response. It publishes `detached:<id>:<detail>` after release and `failed:<id>:<detail>` for service, timeout or consistency errors. The coordinator pauses in a waiting state until the expected acknowledgement arrives; an unavailable bridge cannot become a false successful cycle.

## Metrics and terminal-event ownership

`CellStatus` is transient-local so late observers can recover terminal results. The coordinator owns the completed/failed counts and emits one terminal event per object. The metrics node de-duplicates terminal records and writes either:

- the exact `metrics_csv:=...` launch path; or
- `/tmp/sorting_cell_metrics_<UTC timestamp>.csv` when no path is supplied.

## Verification layers

The two headless terms refer to different scopes:

| Layer | ROS 2 / Gazebo | Purpose |
| --- | --- | --- |
| Pure-Python headless demo | No | Fast deterministic validation of planning, routing and sampled kinematics |
| Server-only ROS 2 end-to-end run | Yes | Camera, DDS, action, measured-feedback, attachment-service and queried Gazebo bin-pose verification without GUI windows |

`tools/verify_ros2_demo.py` independently checks active controllers, camera frames, joint states, all three image-derived detections, detection frames/classes, unique terminal events, no failures, named Gazebo pose query success and each part's final bin bounds. The 2026-07-15 reference payload is stored at [`evidence/ros2_runtime_verification.json`](evidence/ros2_runtime_verification.json).

## Limitations

- HSV classification is suitable for the controlled scene, not a claim of general-purpose industrial recognition.
- Analytical IK and guarded waypoints are appropriate for this fixed workcell; MoveIt 2 is the intended extension for obstacle-rich free-space planning.
- `SetEntityPose` is a simulation adapter, not grasp physics or real gripper control.
- The simulation is not a safety certification and does not replace hardware interlocks, safe torque off or the manufacturer's controller limits.
